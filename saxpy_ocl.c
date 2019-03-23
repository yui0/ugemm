// ©2019 Yuichiro Nakada
// clang -Os saxpy_ocl.c -o saxpy_ocl `pkg-config --libs --cflags OpenCL` -lm
// clang -Os saxpy_ocl.c -o saxpy_ocl -framework opencl
// clang -Os saxpy_ocl.c -o saxpy_ocl -L/opt/amdgpu-pro/lib64/ -lOpenCL
// LD_LIBRARY_PATH=/opt/amdgpu-pro/lib64 ./saxpy_ocl
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

#define PRECISION	32

#define WGS 64     // The local work-group size
#define WPT 1      // The amount of work-per-thread
#define VW 1       // Vector width of vectors X and Y

// Data-type: single or double precision
#if PRECISION == 32
  typedef float real;
  typedef float2 real2;
  typedef float4 real4;
  typedef float8 real8;
  typedef float16 real16;
  #define ZERO 0.0f
#elif PRECISION == 64
  #if __OPENCL_VERSION__ <= CL_VERSION_1_1 // This the default on OpenCL 1.2 or higher
     #pragma OPENCL EXTENSION cl_khr_fp64: enable
  #endif
  typedef double real;
  typedef double2 real2;
  typedef double4 real4;
  typedef double8 real8;
  typedef double16 real16;
  #define ZERO 0.0
#endif

// Data-widths
#if VW == 1
  typedef real realV;
#elif VW == 2
  typedef real2 realV;
#elif VW == 4
  typedef real4 realV;
#elif VW == 8
  typedef real8 realV;
#elif VW == 16
  typedef real16 realV;
#endif

//typedef float real_arg;
//#define GetRealArg(x) (half)x
#define MultiplyAdd(c,a,b) c = mad(a, b, c)

// The vectorized multiply-add function
inline realV MultiplyAddVector(realV cvec, const real aval, const realV bvec) {
  #if VW == 1
    MultiplyAdd(cvec, aval, bvec);
  #elif VW == 2
    MultiplyAdd(cvec.x, aval, bvec.x);
    MultiplyAdd(cvec.y, aval, bvec.y);
  #elif VW == 4
    MultiplyAdd(cvec.x, aval, bvec.x);
    MultiplyAdd(cvec.y, aval, bvec.y);
    MultiplyAdd(cvec.z, aval, bvec.z);
    MultiplyAdd(cvec.w, aval, bvec.w);
  #elif VW == 8
    MultiplyAdd(cvec.s0, aval, bvec.s0);
    MultiplyAdd(cvec.s1, aval, bvec.s1);
    MultiplyAdd(cvec.s2, aval, bvec.s2);
    MultiplyAdd(cvec.s3, aval, bvec.s3);
    MultiplyAdd(cvec.s4, aval, bvec.s4);
    MultiplyAdd(cvec.s5, aval, bvec.s5);
    MultiplyAdd(cvec.s6, aval, bvec.s6);
    MultiplyAdd(cvec.s7, aval, bvec.s7);
  #elif VW == 16
    MultiplyAdd(cvec.s0, aval, bvec.s0);
    MultiplyAdd(cvec.s1, aval, bvec.s1);
    MultiplyAdd(cvec.s2, aval, bvec.s2);
    MultiplyAdd(cvec.s3, aval, bvec.s3);
    MultiplyAdd(cvec.s4, aval, bvec.s4);
    MultiplyAdd(cvec.s5, aval, bvec.s5);
    MultiplyAdd(cvec.s6, aval, bvec.s6);
    MultiplyAdd(cvec.s7, aval, bvec.s7);
    MultiplyAdd(cvec.s8, aval, bvec.s8);
    MultiplyAdd(cvec.s9, aval, bvec.s9);
    MultiplyAdd(cvec.sA, aval, bvec.sA);
    MultiplyAdd(cvec.sB, aval, bvec.sB);
    MultiplyAdd(cvec.sC, aval, bvec.sC);
    MultiplyAdd(cvec.sD, aval, bvec.sD);
    MultiplyAdd(cvec.sE, aval, bvec.sE);
    MultiplyAdd(cvec.sF, aval, bvec.sF);
  #endif
  return cvec;
}

// Full version of the kernel with offsets and strided accesses
/*__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xaxpy(const int n, const real_arg arg_alpha,
	const __global real* restrict xgm, const int x_offset, const int x_inc,
	__global real* ygm, const int y_offset, const int y_inc)
{
	const real alpha = GetRealArg(arg_alpha);

	// Loops over the work that needs to be done (allows for an arbitrary number of threads)
	for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
		real xvalue = xgm[id*x_inc + x_offset];
		MultiplyAdd(ygm[id*y_inc + y_offset], alpha, xvalue);
	}
}*/

// Faster version of the kernel without offsets and strided accesses.
// Also assumes that 'n' is dividable by 'VW', 'WGS' and 'WPT'.
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XaxpyFastest(const int n, const real alpha, const __global realV* restrict xgm, __global realV* ygm)
{
	\n\x23pragma unroll\n
	for (int _w=0; _w<WPT; _w++) {
		const int id = _w*get_global_size(0) + get_global_id(0);
		realV xvalue = xgm[id];
		realV yvalue = ygm[id];
		ygm[id] = MultiplyAddVector(yvalue, alpha, xvalue);
	}
}

);

// Size of the matrices
#define SIZE 1024
int N = SIZE*SIZE;
float ALPHA, X[SIZE*SIZE], Y[SIZE*SIZE], Z[SIZE*SIZE];
args_t args[] = {
	{ 0, sizeof(int), 0, &N, 0 },
	{ 0, sizeof(float), 0, &ALPHA, 0 },
	{ CL_MEM_READ_ONLY,  sizeof(float)*SIZE*SIZE, 0, X, OCL_INPUT },
	{ CL_MEM_READ_WRITE, sizeof(float)*SIZE*SIZE, 0, Y, OCL_OUTPUT },
	{ 0, 0, 0, 0, 0 },
};
ocl_t kernel[] = {
	{ "XaxpyFastest", 0, 1,{WGS},{WGS}, args },
};
int ksz = sizeof(kernel)/sizeof(kernel[0]);

int main()
{
	ALPHA = 0.5;
	for (int i=0; i<SIZE*SIZE; i++) { X[i] = 3.6*i + i*i + 3.1; }
	for (int i=0; i<SIZE*SIZE; i++) { Y[i] = Z[i] = 1.2*i + 0.01*i*i + 13.9; }

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
//	double gflop = ((long)K * (long)M * (long)N * 2) / (1000*1000*1000);
	double gflop = (SIZE*SIZE) / (1000*1000*1000);
	printf(">>> Done: took %.3lf seconds per run, %.1lf GFLOPS\n", runtime, gflop/runtime);

	oclReleaseKernel(kernel, ksz);
	oclFinish();

	for (int i=0; i<SIZE*SIZE; i++) {
		Z[i] += ALPHA * X[i];
	}
	cmp_results(SIZE, SIZE, Z, Y, 1);
}
