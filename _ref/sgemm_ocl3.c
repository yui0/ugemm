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

#define TS 16		// Threadblock sizes
#define WPT 8		// The amount of work-per-thread, i.e. the thread-coarsening factor
#define RTS (TS/WPT)	// The reduced tile-size in one dimension

char kernel_code[] = OCLSTRINGIFY(

// Increased the amount of work-per-thread by a factor WPT
__kernel void gemm(const int M, const int N, const int K,
	const __global float* A, const __global float* B, __global float* C)
{
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Initialise the accumulation registers
    float acc[WPT];
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        for (int w=0; w<WPT; w++) {
            const int tiledRow = TS*t + row;
            const int tiledCol = TS*t + col;
            Asub[col + w*RTS][row] = A[(tiledCol + w*RTS)*M + globalRow];
            Bsub[col + w*RTS][row] = B[(globalCol + w*RTS)*K + tiledRow];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            for (int w=0; w<WPT; w++) {
                acc[w] += Asub[k][row] * Bsub[col + w*RTS][k];
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int w=0; w<WPT; w++) {
        C[(globalCol + w*RTS)*M + globalRow] = acc[w];
    }
}

);

// Size of the matrices - K, M, N (squared)
//#define SIZE 4096
#define SIZE 1024

int M = SIZE;
int N = SIZE;
int K = SIZE;
float A[SIZE*SIZE], B[SIZE*SIZE], C[SIZE*SIZE], Z[SIZE*SIZE];
args_t args[] = {
	{ 0, sizeof(int), 0, &M, 0 },
	{ 0, sizeof(int), 0, &N, 0 },
	{ 0, sizeof(int), 0, &K, 0 },
	{ CL_MEM_READ_ONLY,  sizeof(float)*SIZE*SIZE, 0, A, OCL_INPUT },
	{ CL_MEM_READ_ONLY,  sizeof(float)*SIZE*SIZE, 0, B, OCL_INPUT },
	{ CL_MEM_READ_WRITE, sizeof(float)*SIZE*SIZE, 0, C, OCL_OUTPUT },
	{ 0, 0, 0, 0, 0 },
};
ocl_t kernel[] = {
	{ "gemm", 0, 2,{/*M*/SIZE,/*N*/SIZE/WPT},{TS,TS/WPT}, args },
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
			// Row Major
			/*for (int k=0; k<K; k++) {
				sum += A[k + m * lda] * B[n + k * ldb];
			}
			Z[n + m * ldc] = alpha * sum + beta * Z[n + m * ldc];*/
			// Column Major
			for (int k=0; k<K; k++) {
				sum += A[m + k * lda] * B[k + n * ldb];
			}
			Z[m + n * ldc] = alpha * sum + beta * Z[m + n * ldc];
		}
	}
	cmp_results(M, N, Z, C, ldc);
}
