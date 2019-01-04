// ©2019 Yuichiro Nakada
// clang -Os sgemm_ocl.c -o sgemm_ocl `pkg-config --libs --cflags OpenCL`
// clang -Os sgemm_ocl.c -o sgemm_ocl -framework opencl
// clang -Os sgemm_ocl.c -o sgemm_ocl -L/opt/amdgpu-pro/lib64/ -lOpenCL
// LD_LIBRARY_PATH=/opt/amdgpu-pro/lib64 ./sgemm_ocl
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include "ocl.h"

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

char kernel_code[] = OCLSTRINGIFY(

kernel void gemm1(const int M, const int N, const int K,
	const global float* A, const global float* B, global float* C)
{
	// Thread identifiers
	const int globalRow = get_global_id(0); // Row ID of C (0..M)
	const int globalCol = get_global_id(1); // Col ID of C (0..N)

	// Compute a single element (loop over K)
	float acc = 0.0f;
	for (int k=0; k<K; k++) {
		acc += A[k*M + globalRow] * B[globalCol*K + k];
	}

	// Store the result
	C[globalCol*M + globalRow] = acc;
}

);

// Size of the matrices - K, M, N (squared)
//#define SIZE 4096
#define SIZE 1024
// Threadblock sizes
//#define TS 32
#define TS 16

int M = SIZE;
int N = SIZE;
int K = SIZE;
float A[SIZE*SIZE], B[SIZE*SIZE], C[SIZE*SIZE], Z[SIZE*SIZE];
args_t args[] = {
	{ 0, sizeof(int), 0, &M, 0 },
	{ 0, sizeof(int), 0, &N, 0 },
	{ 0, sizeof(int), 0, &K, 0 },
	/*{ CL_MEM_READ_ONLY,  sizeof(float)*SIZE*SIZE, 0, A, OCL_WRITE },
	{ CL_MEM_READ_ONLY,  sizeof(float)*SIZE*SIZE, 0, B, OCL_WRITE },
	{ CL_MEM_READ_WRITE, sizeof(float)*SIZE*SIZE, 0, C, OCL_READ },*/
	{ CL_MEM_READ_WRITE, sizeof(float)*SIZE*SIZE, 0, A, OCL_READ|OCL_WRITE },
	{ CL_MEM_READ_WRITE, sizeof(float)*SIZE*SIZE, 0, B, OCL_READ|OCL_WRITE },
	{ CL_MEM_READ_WRITE, sizeof(float)*SIZE*SIZE, 0, C, OCL_READ|OCL_WRITE },
	{ 0, 0, 0, 0, 0 },
};
ocl_t kernel[] = {
	{ "gemm1", 0, 2,{/*M*/SIZE,/*N*/SIZE,0,},{TS,TS,0,}, args },
};
int ksz = sizeof(kernel)/sizeof(kernel[0]);

int main()
{
	for (int i=0; i<M*K; i++) { A[i] = 3.6*i + i*i + 3.1; }
	for (int i=0; i<K*N; i++) { B[i] = 1.2*i + 0.01*i*i + 13.9; }
	for (int i=0; i<M*N; i++) { C[i] = 0.0; }
	for (int i=0; i<M*N; i++) { Z[i] = 0.0; }

	oclSetup(0, 0);
	oclKernel(kernel, ksz, "-cl-denorms-are-zero -cl-finite-math-only -cl-fast-relaxed-math -Werror", kernel_code);
	oclKernelArgs(kernel, ksz);

	struct timeval tv;
	struct timezone dummy;
	gettimeofday(&tv, &dummy);
	double starttime = (double)tv.tv_sec + 1.0e-6*((double)tv.tv_usec);

	oclKernelArgsWrite(args);
	oclRun(&kernel[0]);
	oclKernelArgsRead(args);

	gettimeofday(&tv, &dummy);
	double endtime = (double)tv.tv_sec + 1.0e-6*((double)tv.tv_usec);
	double runtime = (endtime - starttime) / (double)/*NUM_RUNS*/1;
	double gflop = ((long)K * (long)M * (long)N * 2) / (1000*1000*1000);
	printf(">>> Done: took %.3lf seconds per run, %.1lf GFLOPS\n", runtime, gflop/runtime);

	oclReleaseKernel(kernel, ksz);
	oclFinish();

	int lda = SIZE;
	int ldb = SIZE;
	int ldc = SIZE;
	float alpha = 1;
	float beta = 0;
	for (int m=0; m<M; m++) {
		for (int n=0; n<N; n++) {
			register float sum = 0.0;
			/*for (int k=0; k<K; k++) {
				sum += A[k + m * lda] * B[n + k * ldb];
			}
			Z[n + m * ldc] = alpha * sum + beta * Z[n + m * ldc];*/
			for (int k=0; k<K; k++) {
				sum += A[m + k * lda] * B[k + n * ldb];
			}
			Z[m + n * ldc] = alpha * sum + beta * Z[m + n * ldc];
		}
	}
	cmp_results(M, N, Z, C, ldc);
	/*for (int i=0; i<SIZE*SIZE; i++) {
		printf("%f + %f = %f\n", x[i], y[i], z[i]);
	}
	printf("\n");*/
}
