typedef float afloat __attribute__ ((__aligned__(256)));
typedef __m256 float8;

#define _load_float8(PTR)	(_mm256_load_ps(PTR))
#define _store_float8(PTR, VAL)	*((float8 *)(PTR)) = (VAL);
#define _add_float8(PTR, VAL)	*((float8 *)(PTR)) += (VAL);
#define _broadcast_float8(VAL)	(_mm256_set1_ps(VAL))
//#define _fma_float8(A, B, C)	(_mm256_fmadd_ps((A), (B), (C)))
#define _fma_float8(A, B, C)	(_mm256_add_ps(_mm256_mul_ps((A), (B)), (C)))

#define _regsA	6
#define _regsB	2

// 6x16 kernel without blocking
// Requires AVX-2 and FMA
// See a full description at:  http://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf
static inline void matmul_dot_inner(
	const afloat * __restrict__ A, const afloat * __restrict__ B, afloat * C,
	const int M, const int N, const int K,
	const int m, const int n
) {
	// Chose kernel size _regsA x (_regsB * 8)
	// - SIMD width is 8*32 (so must be multiple of 8)
	// - Also overall _regsA * _regsB  registers for C
	// - Number of registers depends on AVX, AVX-2 or AVX-512
	// - So for example having 6x16 means 6x2 registers used for C block
	// - This leaves 4 for sections of A and B (needed to do fma)
	// - To use SIMD, need to store in registers
	// - Note: Intel paper uses 30x8
	float8 csum[_regsA][_regsB] = {{_broadcast_float8(0)}}; // Broadcast 32-bit (SP) 0 to all 8 elements

	// This is the same for loop as in naive implementation, except now instead of the k indexing
	// a single dot product of 2 vectors of size k (a row of A and a col of B),
	// the k is indexing @_regsA rows of A and (@_regsB * 8) cols of B
	// Since the SIMD width is 8 (256 bits), need to do _regsA * _regsB fmas here
	for (int k=0; k<K; k++) {
		for (unsigned ai=0; ai<_regsA; ai++) {
			float8 aa = _broadcast_float8(A[(m + ai) * K + k]);
			for (unsigned bi=0; bi<_regsB; bi++) {
				float8 bb = _load_float8(&B[k * N + n + bi * 8]);
				csum[ai][bi] = _fma_float8(aa, bb, csum[ai][bi]);
			}
		}
	}
	// Write registers back to C
	for (unsigned ai=0; ai<_regsA; ai++) {
		for (unsigned bi=0; bi<_regsB; bi++) {
			_store_float8(&C[(m + ai) * N + n + bi * 8], csum[ai][bi]);
		}
	}
}

static inline void matmul_dot_inner_block(
	const afloat * __restrict__ A, const afloat * __restrict__ B, afloat * C,
	const int M, const int N, const int K,
	const int jc, const int nc,
	const int pc, const int kc,
	const int ic, const int mc,
	const int jr, const int nr,
	const int ir, const int mr
) {
	// Chose kernel size _regsA x (_regsB * 8)
	// - SIMD width is 8*32 (so must be multiple of 8)
	// - Also overall _regsA * _regsB  registers for C
	// - Number of registers depends on AVX, AVX-2 or AVX-512
	// - So for example having 6x16 means 6x2 registers used for C block
	// - This leaves 4 for sections of A and B (needed to do fma)
	// - To use SIMD, need to store in registers
	// - Note: Intel paper uses 30x8
	float8 csum[_regsA][_regsB] = {{_broadcast_float8(0)}}; // Broadcast 32-bit (SP) 0 to all 8 elements

	// This is the same for loop as in naive implementation, except now instead of the k indexing
	// a single dot product of 2 vectors of size k (a row of A and a col of B),
	// the k is indexing @_regsA rows of A and (@_regsB * 8) cols of B
	// Since the SIMD width is 8 (256 bits), need to do _regsA * _regsB fmas here
	for (int k = 0; k < kc; k++) {
		for (unsigned ai = 0; ai < _regsA; ai++) {
			float8 aa = _broadcast_float8(A[(ic + ir + ai) * K + pc + k]);
			for (unsigned bi = 0; bi < _regsB; bi++) {
				float8 bb = _load_float8(&B[(pc + k) * N + jc + jr + bi * 8]);
				csum[ai][bi] = _fma_float8(aa, bb, csum[ai][bi]);
			}
		}
	}
	// Write registers back to C
	for (unsigned ai = 0; ai < _regsA; ai++) {
		for (unsigned bi = 0; bi < _regsB; bi++) {
			_add_float8(&C[(ic + ir + ai) * N + jc + jr + bi * 8], csum[ai][bi]);
		}
	}
}

static inline void sgemm(const int M, const int N, const int K, const float *A, const float *B, float *C)
{
//	bool transpose_A = false;
//	bool transpose_B = false;
//	assert_sgemm_parameters(/*Order=*/CblasRowMajor, /*TransA=*/CblasNoTrans, /*TransB=*/CblasNoTrans, N, M, K, K, N, N, transpose_A, transpose_B);
//	assert(!transpose_A);
//	assert(!transpose_B);

	// kc * 16 fits in L1, which is 32 K
	// kc * mc fits in L2, which is 256 K
	// kc * nc fits in L3, which is 4 M
	const int nc = N;
	const int kc = 240;
	const int mc = 120;
	const int nr = 2 * 8;
	const int mr = 6;

	//omp_set_num_threads(8);

	// Usually 1 iteration, cannot parallelize
	for (int jc=0; jc<N; jc+=nc) {
		// 8 iterations, not worth parallelizing
		for (int pc=0; pc<K; pc+=kc) {
			// 16 iterations, not worth parallelizing
			for (int ic=0; ic<M; ic+=mc) {
				// 120 iterations, worth parallelizing
				#pragma omp parallel for
				for (int jr=0; jr<nc; jr+=nr) {
					// 20 iterations, not worth parallelizing
					for (int ir=0; ir<mc; ir+=mr) {
						matmul_dot_inner_block(A, B, C, M, N, K, jc, nc, pc, kc, ic, mc, jr, nr, ir, mr);
					}
				}
			}
		}
	}
}
