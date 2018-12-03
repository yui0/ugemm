// clang -Ofast -o check_sgemm check_sgemm.c -mavx

#include "ugemm.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#if defined(__i386__)
static __inline__ unsigned long long rdtsc(void)
{
    unsigned long long int x;
    __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
    return x;
}
#elif defined(__x86_64__)
static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
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

float *random_matrix(int rows, int cols)
{
	float *m = malloc_a(rows*cols*sizeof(float), 32);
	for (int i=0; i<rows*cols; i++) {
		m[i] = (float)rand()/RAND_MAX;
	}
	return m;
}

static void cmp_results(int M, int N, const float *ref, const float *res, int ld)
{
	double maxErr = 0;
	double s2Err = 0;
	double s1Ref = 0;
	double s2Ref = 0;
	int maxI = 0;
	for (int m = 0; m < M; ++m) {
		for (int n = 0; n < N; ++n) {
			double refV = ref[m*ld+n];
			double resV = res[m*ld+n];
			double err  = resV - refV;
			if (maxErr < fabs(err)) {
				maxErr = fabs(err);
				maxI = m*ld+n;
			}
			s2Err += err*err;
			s1Ref += refV;
			s2Ref += refV*refV;
		}
	}
	double stdErr = sqrt(s2Err / (M*N));
	double stdRef = sqrt(s2Ref*(M*N) - s1Ref*s1Ref)/((M*N));
	printf("%.3e/%.3e=%.3e. %.3e at [%3d,%3d] %18.10e vs %18.10e %s\n",
		stdErr, stdRef, stdErr/stdRef,
		maxErr, maxI/ld, maxI%ld,
		(double)ref[maxI], (double)res[maxI],
		maxErr > stdRef*1e-5 ? "FAIL !!!" : (maxErr > stdRef*3e-5 || stdErr > stdRef*1e-6 ? "Sucks !" : "")
	);
}

static void test_sgemm(
        int M, int N, int K,
        float alpha,
        const float *A, int lda,
        const float *B, int ldb,
        float beta,
        float *C, int ldc,
        int nIter,
        const float *srcC,
        void (*uut)(
		char major, char transa, char transb,
                int M, int N, int K,
                float alpha,
                const float *A, int lda,
                const float *B, int ldb,
                float beta,
                float *C, int ldc)
)
{
	for (int i=0; i<nIter*M*ldc; ++i) {
		C[i] = srcC[i];
	}

	uint64_t dt[nIter];
	for (int it=0; it<nIter; ++it) {
		uint64_t t0 = __rdtsc();
		uut(
			'R', 'N', 'N',
			M, N, K,
			alpha,
			&A[it*M*lda], lda,
			&B[it*K*ldb], ldb,
			beta,
			&C[it*M*ldc], ldc
		);
		uint64_t t1 = __rdtsc();
		dt[it] = t1-t0;
	}
	double a = 0;
	for (int it=0; it<nIter; ++it) {
		printf(" %.0f", (double)dt[it]);
		a += dt[it];
	}
	a /= nIter;
//	printf(": med %lu. %.3f FLOP/clk\n", dt[nIter/2], M*N*K*2.0/(dt[nIter/2]));
	printf(": med %.0f. %.3f FLOP/clk\n", a, M*N*K*2.0/a);
	print_GFLOPS(M*N*K*2.0/a, 1);

	float __attribute__((aligned(32))) refC[M*ldc];
	for (int it=0; it<nIter; it++) {
		for (int i=0; i<M*ldc; i++) {
			refC[i] = srcC[it*M*ldc+i];
		}
		sgemm_cpu('R', 'N', 'N',
		        M, N, K, alpha, &A[it*M*lda], lda, &B[it*K*ldb], ldb, beta, refC, ldc);
		cmp_results(M, N, refC, &C[it*M*ldc], ldc);
	}
}

int main(int argz, char** argv)
{
//	int M = 128; // RNN
//	int N = 361; // RNN
//	int K = 1152;
	int M = 500;
	int N = 500;
	int K = 500;
	float alpha = 1;
	float beta  = 0;
	int lda = 0;
	int ldb = 0;
	int ldc = 0;

	for (int arg_i = 1; arg_i < argz; ++arg_i) {
		char* arg = argv[arg_i];
		static const char* prefTab[] = {
			"alpha", "beta", "M", "N", "K", "lda", "ldb", "ldc"
		};
		for (int pref_i = 0; pref_i < sizeof(prefTab)/sizeof(prefTab[0]); ++pref_i) {
			const char* pref = prefTab[pref_i];
			size_t preflen = strlen(pref);
			if (strncasecmp(pref, arg, preflen)==0 && arg[preflen]=='=') {
				if (pref_i < 2) {
					// floating point arguments
					char* endp;
					double val = strtod(&arg[preflen+1], &endp);
					if (endp==&arg[preflen+1]) {
						fprintf(stderr, "Bad parameter '%s'. '%s' is not a number.\n", arg, &arg[preflen+1]);
						return 1;
					}
					switch (pref_i) {
					case 0:
						alpha = (float)val;
						break;
					case 1:
						beta = (float)val;
						break;
					default:
						break;
					}
				} else {
					// integer arguments
					char* endp;
					long val = strtol(&arg[preflen+1], &endp, 0);
					if (endp==&arg[preflen+1] || val <= 0) {
						fprintf(stderr, "Bad parameter '%s'. '%s' is not a positive number.\n", arg, &arg[preflen+1]);
						return 1;
					}
					switch (pref_i) {
					case 2:
						M = val;
						break;
					case 3:
						N = val;
						break;
					case 4:
						K = val;
						break;
					case 5:
						lda = val;
						break;
					case 6:
						ldb = val;
						break;
					case 7:
						ldc = val;
						break;
					default:
						break;
					}
				}
				goto next_arg;
			}
		}
next_arg:;
	}

	if (lda == 0) lda = K;
	if (ldb == 0) ldb = N;
	if (ldc == 0) ldc = N;
	if (lda < K) {
		fprintf(stderr, "Bad parameter lda=%d. Should be greater or equal to K=%d\n", lda, K);
		return 1;
	}
	if (ldb < N) {
		fprintf(stderr, "Bad parameter ldb=%d. Should be greater or equal to N=%d\n", ldb, N);
		return 1;
	}
	if (ldc < N) {
		fprintf(stderr, "Bad parameter ldc=%d. Should be greater or equal to N=%d\n", ldc, N);
		return 1;
	}

	printf("Running SGEMM with M=%d, N=%d, K=%d, alpha=%f, lda=%d, ldb=%d, beta=%f, ldc=%d\n",
	       M, N, K, alpha, lda, ldb, beta, ldc);


	const int nIter = 11;
	float *a = random_matrix(nIter*M, lda);
	float *b = random_matrix(nIter*K, ldb);
	float *c = random_matrix(nIter*M, ldc);
	float *sc = random_matrix(nIter*M, ldc);
	test_sgemm(M, N, K, alpha, a, lda, b, ldb, beta, c, ldc, nIter, sc, sgemm_cpu);
	test_sgemm(M, N, K, alpha, a, lda, b, ldb, beta, c, ldc, nIter, sc, sgemm_avx);
	free_a(sc);
	free_a(c);
	free_a(b);
	free_a(a);

	return 0;
}
