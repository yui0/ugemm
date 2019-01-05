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

#define TSM 128                // The tile-size in dimension M
#define TSN 128                // The tile-size in dimension N
#define TSK 16                 // The tile-size in dimension K
#define WPTM 8                 // The work-per-thread in dimension M
#define WPTN 8                 // The work-per-thread in dimension N
#define RTSM (TSM/WPTM)        // The reduced tile-size in dimension M
#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
#define LPTA ((TSK*TSM)/(RTSM*RTSN)) // Loads-per-thread for A
#define LPTB ((TSK*TSN)/(RTSM*RTSN)) // Loads-per-thread for B

char kernel_code[] = OCLSTRINGIFY(

// Use 2D register blocking (further increase in work per thread)
__kernel void gemm(const int M, const int N, const int K,
	const __global float* A, const __global float* B, __global float* C)
{
    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM == RTSM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN == RTSN)
    const int offsetM = TSM*get_group_id(0); // Work-group offset
    const int offsetN = TSN*get_group_id(1); // Work-group offset

    // Local memory to fit a tile of A and B
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK+2];

    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    // Initialise the accumulation registers
    \n\x23pragma unroll\n
    for (int wm=0; wm<WPTM; wm++) {
        \n\x23pragma unroll\n
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }
    
    // Loop over all tiles
    const int numTiles = K/TSK;
    int t=0;
    do {
        // Load one tile of A and B into local memory
        \n\x23pragma unroll\n
        for (int la=0; la<LPTA; la++) {
            int tid = tidn*RTSM + tidm;
            int id = la*RTSN*RTSM + tid;
            int row = id % TSM; //MOD2(id,TSM);
            int col = id / TSM; //DIV2(id,TSM);
            int tiledIndex = TSK*t + col;
            Asub[col][row] = A[tiledIndex*M + offsetM + row];
            Bsub[row][col] = B[tiledIndex*N + offsetN + row];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        for (int k=0; k<TSK; k++) {

            // Cache the values of Bsub in registers
            \n\x23pragma unroll\n
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[col][k];
            }

            // Perform the computation
            \n\x23pragma unroll\n
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[k][row];
                \n\x23pragma unroll\n
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);

        // Next tile
        t++;
    } while (t<numTiles);

    // Store the final results in C
    \n\x23pragma unroll\n
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        \n\x23pragma unroll\n
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
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
	{ CL_MEM_READ_ONLY,  sizeof(float)*SIZE*SIZE, 0, A, OCL_INPUT },
	{ CL_MEM_READ_ONLY,  sizeof(float)*SIZE*SIZE, 0, B, OCL_INPUT },
	{ CL_MEM_READ_WRITE, sizeof(float)*SIZE*SIZE, 0, C, OCL_OUTPUT },
	{ 0, 0, 0, 0, 0 },
};
ocl_t kernel[] = {
	{ "gemm", 0, 2,{/*M*/SIZE/WPTM,/*N*/SIZE/WPTN,},{TSM/WPTM,TSN/WPTN}, args },
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
