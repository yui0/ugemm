// ©2020-2021 Yuichiro Nakada
// clang -Os sgemm_gl.c -o sgemm_gl `pkg-config --libs --cflags gl egl gbm` -lglfw -lm
// dnf install mesa-libgbm-devel libdrm-devel mesa-libGL-devel mesa-libGLU-devel mesa-libEGL-devel mesa-libGLES-devel glfw-
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#ifdef CATS_OPENGL
#include "sgemm_gl1.h" // 46 GFLOPS (69)
//#include "sgemm_gl2.h" // 58 GFLOPS
//#include "sgemm_gl3.h" // 63 GFLOPS (82)
//#include "sgemm_gl44.h" // 75 GFLOPS
//#include "sgemm_gl48.h" // 70 GFLOPS
//#include "sgemm_gl5.h" // 67 GFLOPS
//#include "sgemm_gl7.h" // 85 GFLOPS
//#include "sgemm_gl.h" // 90 GFLOPS

#define sgemm_init(s1, s2, s3)				sgemm_gl_init(s1*sizeof(float), s2*sizeof(float), s3*sizeof(float))
#define sgemm_finish()					sgemm_gl_finish()
#define sgemm_rnn(M, N, K, ALPHA, A, B, BETA, C)	sgemm_gl(GEMM1_RNN, M, N, K, ALPHA, A, B, BETA, C)
#define sgemm_rnt(M, N, K, ALPHA, A, B, BETA, C)	sgemm_gl(GEMM1_RNT, M, N, K, ALPHA, A, B, BETA, C)
#define sgemm_rtn(M, N, K, ALPHA, A, B, BETA, C)	sgemm_gl(GEMM1_RTN, M, N, K, ALPHA, A, B, BETA, C)
#else
//#include "sgemm_ocl1.h" // 30 GFLOPS
#include "sgemm_ocl2.h" // 200 GFLOPS
//#include "sgemm_ocl.h" // 30 GFLOPS

#define sgemm_init(s1, s2, s3)				sgemm_ocl_init(0, 0, (s1+s2+s3)*10*sizeof(float))
#define sgemm_finish()					sgemm_ocl_finish()
#define sgemm_rnn(M, N, K, ALPHA, A, B, BETA, C)	sgemm_ocl('N', 'N', M, N, K, ALPHA, A, B, BETA, C)
#define sgemm_rnt(M, N, K, ALPHA, A, B, BETA, C)	sgemm_ocl('N', 'T', M, N, K, ALPHA, A, B, BETA, C)
#define sgemm_rtn(M, N, K, ALPHA, A, B, BETA, C)	sgemm_ocl('T', 'N', M, N, K, ALPHA, A, B, BETA, C)
#endif

#ifndef real
#define real		float
#endif

inline void gemm_rnn(int M, int N, int K, real alpha, real *A, real *B, real beta, real *C)
{
	if (beta==0.0) {
		memset(C, 0, M*N*sizeof(real));
	} else if (beta!=1.0) {
		for (int i=0; i<M*N; i++) C[i] *= beta;
	}
/*	const int lda = K;
	const int ldb = N;
	const int ldc = N;*/
	#pragma omp parallel for
	for (int m=0; m<M; ++m) { // fast
		for (int k=0; k<K; ++k) {
			register real A_PART = alpha * A[m*K+k];
			for (int n=0; n<N; ++n) {
				C[m*N+n] += A_PART * B[k*N+n];
			}
		}
	}
}

inline void gemm_rnt(int M, int N, int K, real alpha, real *A, real *B, real beta, real *C)
{
	if (beta==0.0) {
		memset(C, 0, M*N*sizeof(real));
	} else if (beta!=1.0) {
		for (int i=0; i<M*N; i++) C[i] *= beta;
	}
/*	const int lda = K;
	const int ldb = K;
	const int ldc = N;*/
	#pragma omp parallel for
	for (int m=0; m<M; ++m) {
		for (int n=0; n<N; ++n) {
			register real sum = 0;
			for (int k=0; k<K; ++k) {
				sum += A[m*K+k] * B[k+K*n];
//				sum += A[m*K+k] * (*B++);
			}
			C[m*N+n] += alpha * sum;
//			(*C++) = alpha * sum;
		}
	}
}

inline void gemm_rtn(int M, int N, int K, real alpha, real *A, real *B, real beta, real *C)
{
	if (beta==0.0) {
		memset(C, 0, M*N*sizeof(real));
	} else if (beta!=1.0) {
		for (int i=0; i<M*N; i++) C[i] *= beta;
	}
/*	const int lda = M;
	const int ldb = N;
	const int ldc = N;*/
	#pragma omp parallel for
	for (int m=0; m<M; ++m) {
		for (int k=0; k<K; ++k) {
			register real A_PART = alpha * A[m+M*k];
			for (int n=0; n<N; ++n) {
				C[m*N+n] += A_PART * B[k*N+n];
			}
		}
	}
}

static void cmp_results(int M, int N, const float *ref, const float *res, int ld)
{
	double maxErr = 0;
	double s2Err = 0;
	double s1Ref = 0;
	double s2Ref = 0;
	int maxI = 0;
	for (int m=0; m<M; ++m) {
		for (int n=0; n<N; ++n) {
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

void print_matrix(float *mat, int m, int n, char N)
{
	printf(">> %c\n", N);
	for (int c=0; c<m; c++) {
		for (int d=0; d<n; d++) {
			printf("%.1f ", mat[c*n+d]);
		}
		printf("\n");
	}
	printf("\n");
}

static struct timeval stv;
/*inline*/ void start()
{
	struct timezone dummy;
	gettimeofday(&stv, &dummy);
}
inline void end(int s, int times)
{
	struct timeval tv;
	struct timezone dummy;
	gettimeofday(&tv, &dummy);
	double starttime = (double)stv.tv_sec + 1.0e-6*((double)stv.tv_usec);
	double endtime = (double)tv.tv_sec + 1.0e-6*((double)tv.tv_usec);
	double runtime = (endtime - starttime) / (double)times;
	double gflop = (s * 2) / (1000*1000*1000);
	printf(">>> Done: took %.3lf seconds per run, %.1lf GFLOPS\n", runtime, gflop/runtime);
}

// Size of the matrices - K, M, N
#define MSIZE 1023
//#define NSIZE 1023
#define NSIZE 1000
#define KSIZE 1023
#define TIMES 20

/*#define PRINT_MAT
#define MSIZE 17
#define NSIZE 17
#define KSIZE 17*/
/*#define MSIZE 16
#define NSIZE 16
#define KSIZE 16*/

/*#define PRINT_MAT
#define MSIZE 3
#define NSIZE 3
#define KSIZE 2*/
/* RNN
A
1.00 2.00
3.00 4.00
5.00 6.00

B
1.00 2.00 3.00
4.00 5.00 6.00

C := A * B
9.00 12.00 15.00
19.00 26.00 33.00
29.00 40.00 51.00
*/

int main(int argc, char* argv[])
{
	const int M = MSIZE;
	const int N = NSIZE;
	const int K = KSIZE;

	sgemm_init(M*K, K*N, M*N);
	printf("\n");

#ifdef OPENCL_SVM
	float *A = _args[0].s;
	float *B = _args[0].s +MSIZE*KSIZE;
	float *C = _args[0].s +MSIZE*KSIZE +KSIZE*NSIZE;
	float *Z = _args[0].s +MSIZE*KSIZE +KSIZE*NSIZE +MSIZE*NSIZE;
#else
	static float A[MSIZE*KSIZE], B[KSIZE*NSIZE], C[MSIZE*NSIZE], Z[MSIZE*NSIZE];
#endif
	for (int i=0; i<M*K; i++) { A[i] = /*3.6*i + i*i + 3.1*//*2*/i+1; }
	for (int i=0; i<K*N; i++) { B[i] = /*1.2*i + 0.01*i*i + 13.9*//*4*/i+1; }
	for (int i=0; i<M*N; i++) { C[i] = 1.0; }
	for (int i=0; i<M*N; i++) { Z[i] = 1.0; }

#ifdef PRINT_MAT
	print_matrix(A, M, K, 'K');
	print_matrix(B, K, N, 'N');
#endif

	printf("GEMM1_RNN: %d %d %d\n", M, N, K);
	start();
	for (int i=0; i<TIMES; i++) sgemm_rnn(M, N, K, 1.0, A, B, 0.0, C);
	end(K*M*N, TIMES);
	gemm_rnn(M, N, K, 1.0, A, B, 0.0, Z);
	cmp_results(M, N, Z, C, /*ldc*/N);
	printf("\n");
#ifdef PRINT_MAT
	print_matrix(C, M, N, 'N');
	print_matrix(Z, M, N, 'N');
#endif

	printf("GEMM1_RNT: %d %d %d\n", M, N, K);
	start();
	for (int i=0; i<TIMES; i++) sgemm_rnt(M, N, K, 1.0, A, B, 0.0, C);
	end(K*M*N, TIMES);
	gemm_rnt(M, N, K, 1.0, A, B, 0.0, Z);
	cmp_results(M, N, Z, C, /*ldc*/N);
	printf("\n");
#ifdef PRINT_MAT
	print_matrix(C, M, N, 'N');
	print_matrix(Z, M, N, 'N');
#endif

	printf("GEMM1_RTN: %d %d %d\n", M, N, K);
	start();
	for (int i=0; i<TIMES; i++) sgemm_rtn(M, N, K, 1.0, A, B, 0.0, C);
	end(K*M*N, TIMES);
	gemm_rtn(M, N, K, 1.0, A, B, 0.0, Z);
	cmp_results(M, N, Z, C, /*ldc*/N);
	printf("\n");
#ifdef PRINT_MAT
	print_matrix(C, M, N, 'N');
	print_matrix(Z, M, N, 'N');
#endif

	sgemm_finish();
}
