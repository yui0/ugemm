// OMP_NUM_THREADS=8 gcc -fopenmp -Ofast -march=native -mavx -funroll-loops matmul.c -o matmul
// https://github.com/HazyResearch/blocking-tutorial/tree/master

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#define ALIGN		(256)
#if 0
#if defined(_MSC_VER) || defined(__MINGW32__)
#define malloc(size)	_aligned_malloc(size, ALIGN)
#define free(p)		_aligned_free(p)
#else
//#define malloc(size)	({ void* p; posix_memalign((void**) &p, ALIGN, (size))==0 ? p : NULL; })
//#define malloc(size)	aligned_alloc(ALIGN, ((size)+(ALIGN)-1) / (ALIGN)*(ALIGN))
#define malloc(size)	aligned_alloc(256, ((size)+(256)-1) / (256)*(256))
#define free(p)		free(p)
#endif  /* _MSC_VER */
#define calloc(n, size)	({ uint64_t s = n * size; void* p = malloc(s); memset(p, 0, s)!=0 ? p : NULL; })
#endif
//#define malloc(sz)	_mm_alloc((256), (sz))
//#define free(ptr)	_mm_free((ptr))
void *amalloc(int size)
{
	void* p;
	posix_memalign((void**)&p, ALIGN, (size));
	return p;
}

#include <immintrin.h>
#include "matrix_kernel_vectorized.c"

const int REPEAT = 1;
//typedef float afloat __attribute__ ((__aligned__(256)));

#define SGEMM_FN sgemm_simd_block_parallel

static inline void sgemm_simd_block_parallel
(
	const int M,
	const int N,
	const int K,
	const float *A,
	const float *B,
	float *C
)
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

	omp_set_num_threads(8);

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
						//matmul_dot_inner_block<6, 2>(A,B,C,M,N,K,jc,nc,pc,kc,ic,mc,jr,nr,ir,mr);
						matmul_dot_inner_block(A,B,C,M,N,K,jc,nc,pc,kc,ic,mc,jr,nr,ir,mr);
					}
				}
			}
		}
	}
}

static inline void sgemm_naive
(
	const int M,
	const int N,
	const int K,
	const float *A,                       // m x k (after transpose if TransA)
	const float *B,                       // k x n (after transpose if TransB)
	float *C                              // m x n
)
{
//	bool transpose_A = false;
//	bool transpose_B = false;
//	assert_sgemm_parameters(/*Order=*/CblasRowMajor, /*TransA=*/CblasNoTrans, /*TransB=*/CblasNoTrans, N, M, K, K, N, N, transpose_A, transpose_B);
	#pragma omp parallel for
	for (int m=0; m<M; ++m) {
		for (int n=0; n<N; ++n) {
			C[m*N + n] = 0;
			for (int k=0; k<K; ++k) {
				size_t A_idx = 0, B_idx = 0;
				A_idx = m*K + k; // A is m x k
				B_idx = n + k*N; // B is k x n
				C[m*N + n] += A[A_idx] * B[B_idx];
			}
		}
	}
}

#if defined(__i386__)
static __inline__ uint64_t rdtsc(void)
{
	uint64_t x;
	__asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
	return x;
}
#elif defined(__x86_64__)
static __inline__ uint64_t rdtsc(void)
{
	unsigned hi, lo;
	__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
	return ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );
}
#endif
/*void cpuid(int param, unsigned int *eax, unsigned int *ebx, unsigned int *ecx, unsigned int *edx)
{
	__asm__( "cpuid"
		: "=a" (*eax), "=b" (*ebx), "=c" (*ecx), "=d" (*edx)
		: "0" (param) );
}*/
// http://laysakura.hateblo.jp/entry/20120106/1325881074
#define GHz 3.50
void print_GFLOPS(double flops, uint64_t cycles)
{
	double GFLOPS = flops * GHz / cycles;
	double sec = (double)cycles * 1e-9 / GHz;
	printf("GFLOPS @ %.2fGHz:\n  %.3f [flops/clock] = %.3f [GFLOPS]  (%.0f flops in %lu clock = %f sec)\n",
		GHz, flops / (double)cycles, GFLOPS, flops, cycles, sec);
}
/*void print_throughput(uint64_t instructions, uint64_t cycles)
{
	printf("Throughput:\n  %.3f [instructions/clock]   (%lu instrucions in %lu clock)\n",
		(double)instructions / (double)cycles, instructions, cycles);
}*/

const int n = 16*6*30;  // 16*6*10 (1 million in N^2), 16*6*30 (8 million in N^2)
const int m = n;
const int k = n;

int main(int argc, char** argv)
{
/*	float x[m*k] __attribute__((aligned(256)));
	float xr[m*k] __attribute__((aligned(256)));
	float y[k*n] __attribute__((aligned(256)));
	float out1[m*n]  __attribute__((aligned(256)));
	float out2[m*n]  __attribute__((aligned(256)));*/

	float *x = amalloc(sizeof(float)*m*k);
	float *xr = amalloc(sizeof(float)*m*k);
	float *y = amalloc(sizeof(float)*k*n);
	float *out1 = amalloc(sizeof(float)*m*n);
	float *out2 = amalloc(sizeof(float)*m*n);

/*	float *x = _mm_alloc(256, sizeof(float)*m*k);
	float *xr = _mm_alloc(256, sizeof(float)*m*k);
	float *y = _mm_alloc(256, sizeof(float)*k*n);
	float *out1 = _mm_alloc(256, sizeof(float)*m*n);
	float *out2 = _mm_alloc(256, sizeof(float)*m*n);*/

	// Generate random data
	//srand((unsigned int)time(NULL));
	srand((unsigned int)0x100);
	printf("Building Matrix: ");
	for (int i=0; i<m; i++) {
		for(int j=0; j<k; j++) {
			x[i*k+j] = (float)(rand()%100) / 100.0;//drand48();
			xr[j*k+i] = x[i*k+j];
		}
	}
	for (int i=0; i<k; i++) {
		for(int j=0; j<n; j++) {
			y[i*n+j] = (float)(rand()%100) / 100.0;//drand48();
		}
	}
	for (int i=0; i<m; i++){
		for(int j=0; j<n; j++) {
			out1[i*n+j] = 0.0;
			out2[i*n+j] = 0.0;
		}
	}
	printf("Done.\n");

	{
		uint64_t t0 = __rdtsc();
		for (int i=0; i<REPEAT; i++) {
			SGEMM_FN(m, n, k, x, y, out1);
		}
		uint64_t t1 = __rdtsc();
		double mtime = (t1-t0) *1e-6 / GHz;
		printf("      Simd block parallel GEMM elapsed time: %f ms, GFlops=%f\n", mtime, ((float) REPEAT*2*m*n*k)/(mtime*1e6));
	}
	{
		uint64_t t0 = __rdtsc();
		for (int i=0; i<REPEAT; i++) {
			sgemm_naive(m, n, k, x, y, out2);
		}
		uint64_t t1 = __rdtsc();
		double mtime = (t1-t0) *1e-6 / GHz;
		printf("      Native GEMM elapsed time: %f ms, GFlops=%f\n", mtime, ((float) REPEAT*2*m*n*k)/(mtime*1e6));
	}

	// Compare outputs
	printf("Computing diff...\n");
	float diff = 0.0;
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++) {
			float u = (out1[i*n+j] - out2[i*n+j]);
			diff += u*u;
		}
	}
	printf("\tNorm Squared=%f\n", diff);
	int nZeros = 0;
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++) {
			if(out1[i*n+j] == 0.0) { nZeros++; }
		}
	}
	printf("\tZeros=%d\n", nZeros);
}
