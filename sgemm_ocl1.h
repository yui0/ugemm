/* public domain Simple, Minimalistic, Fast GEMM library
 *	©2019-2021 Yuichiro Nakada
 *
 * Basic usage:
 *	sgemm_ocl_init(platform, device, max_buffer_size);
 *	sgemm_ocl('N', 'N', M, N, K, A, B, C);
 *	sgemm_ocl_finish();
 * */

#include "ocl.h"

char sgemm_kcode[] = OCLSTRINGIFY(

__kernel void gemm_rnn(__global float* restrict gm, const int8 _info, const float4 _param)
{
	const int M = _info.s0;
	const int N = _info.s1;
	const int K = _info.s2;
	__global float* restrict A = (__global float* restrict)(gm + _info.s3);
	__global float* restrict B = (__global float* restrict)(gm + _info.s4);
	__global float* restrict C = (__global float* restrict)(gm + _info.s5);

	// Thread identifiers
	const int globalRow = get_global_id(0); // Row ID of C (0..M)
	const int globalCol = get_global_id(1); // Col ID of C (0..N)

	if (globalRow >= M || globalCol >= N) return;

	// Compute a single element (loop over K)
	float acc = 0.0f;
	for (int k=0; k<K; k++) {
		acc += A[k + globalRow*K] * B[globalCol + N*k]; // RNN
	}

	// Store the result
	float z = _param.s1;
	if (z) z *= C[globalCol + globalRow*N];
	C[globalCol + globalRow*N] = _param.s0 * acc + z; // Row major
//	C[globalCol + globalRow*N] = acc; // Row major
//	C[globalCol + globalRow*N] = _param.s0 * acc + _param.s1 * C[globalCol + globalRow*N]; // Row major
}

#define TRANSPOSEX 16
#define TRANSPOSEY 16
// Simple transpose kernel for a P * Q matrix
__kernel void transpose(__global float* gm, const int8 _info, const float4 _param)
{
	const int P = _info.s0;
	const int Q = _info.s1;
	__global float* input = (__global float*)(gm + _info.s2);
	__global float* output = (__global float*)(gm + _info.s3);

	// Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0)*TRANSPOSEX + tx; // 0..P
	const int ID1 = get_group_id(1)*TRANSPOSEY + ty; // 0..Q

	// Set-up the local memory for shuffling
	__local float buffer[TRANSPOSEX][TRANSPOSEY];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (ID0 < P && ID1 < Q) {
		buffer[ty][tx] = input[ID1*P + ID0];
	}

	// Synchronise all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// We don't have to swap the x and y thread indices here,
	// because that's already done in the local memory
	const int newID0 = get_group_id(1)*TRANSPOSEY + tx;
	const int newID1 = get_group_id(0)*TRANSPOSEX + ty;

	// Store the transposed result (coalesced)
	if (newID0 < Q && newID1 < P) {
		output[newID1*Q + newID0] = buffer[tx][ty];
	}
}

__kernel void im2col(__global float* gm, const int8 _info, const float4 _param)
{
	__global float* im_src = (__global float*)(gm + _info.s0);
	int channels   = _info.s1;
	int height_inp = _info.s2;
	int width_inp  = _info.s3;
	int kernel_h   = _info.s4;
	int kernel_w   = _info.s4;
	int pad_h      = _info.s5;
	int pad_w      = _info.s5;
	int stride_h   = _info.s6;
	int stride_w   = _info.s6;
	__global float* im_col = (__global float*)(gm + _info.s7);
	int height_out = (height_inp + 2 * pad_h - kernel_h) / stride_h + 1;
	int width_out  = (width_inp + 2 * pad_w - kernel_w) / stride_w + 1;

	int index = get_global_id(0);
	if (index >= height_out * width_out * channels) return;

	int j_out = index % width_out;
	int i_out = (index / width_out) % height_out;
	int c_inp = (index / width_out) / height_out;

	int c_out = c_inp * kernel_h * kernel_w;
	int i_inp = i_out * stride_h - pad_h;
	int j_inp = j_out * stride_w - pad_w;

	im_src += (c_inp * height_inp + i_inp) * width_inp + j_inp;
	im_col += (c_out * height_out + i_out) * width_out + j_out;

	for (int ki = 0; ki < kernel_h; ++ki) {
		for (int kj = 0; kj < kernel_w; ++kj) {
			int i = i_inp + ki;
			int j = j_inp + kj;
			*im_col = (i >= 0 && j >= 0 && i < height_inp && j < width_inp) ? im_src[ki * width_inp + kj] : 0;
			im_col += height_out * width_out;
		}
	}
}

);

void transpose(float *matrix, int m, int n, float *transpose)
{
	for (int c=0; c<m; c++)
		for (int d=0; d<n; d++)
			transpose[d*m+c] = matrix[c*n+d];
}

// Threadblock sizes
#define TS	16

//#define OPENCL_SVM
int _info[8];
float _param[4];
args_t _args[] = {
#ifdef OPENCL_SVM
	{ CL_MEM_READ_WRITE|CL_MEM_SVM_FINE_GRAIN_BUFFER, 0, 0, OCL_SVM },
#else
	{ CL_MEM_READ_WRITE, 0, 0, OCL_BUFFER },
#endif
	{ 0, sizeof(int)*8, _info },
	{ 0, sizeof(float)*4, _param },
	{ 0, 0, 0, 0, 0 },
};
ocl_t _kernel[] = {
	// global: m*MDIMC/MWG, n*NDIMC/NWG
	{ _args, "gemm_rnn", 0, 2,{TS,TS} },

	// global: k, n
	{ _args, "transpose", 0, 2,{TRANSPOSEX,TRANSPOSEY} },
	{ _args, "im2col", 0, 1,{16} },
};
int _ksz = sizeof(_kernel)/sizeof(_kernel[0]);
#define KGEMM_RNN	_kernel[0]
#define KTRANSPOSE	_kernel[1]
#define KIM2COL		_kernel[2]

void sgemm_ocl_init(int platform, int device, size_t size)
{
	_args[0].size = size;

	oclSetup(platform, device);
	oclKernel(_kernel, _ksz, "-cl-denorms-are-zero -cl-finite-math-only -cl-fast-relaxed-math -Werror", sgemm_kcode);
	oclKernelArgs(_kernel, _ksz);
}
static inline void sgemm_ocl(char ta, char tb, int m, int n, int k, float alpha, float *a, float *b, float beta, float *c)
{
	int mk = m*k;
	int kn = k*n;
	int mn = m*n;
	int off_a = 0;
	int off_b = mk;

/*	if (ta=='T') {
		for (int c=0; c<k; c++) {
			for (int d=0; d<m; d++) {
				printf("%.2f ", a[c*m+d]);
			}
			printf("\n");
		}
		printf("\n");

		float t[mk];
		transpose(a, k, m, t);
		a = t;

		for (int c=0; c<m; c++) {
			for (int d=0; d<k; d++) {
				printf("%.2f ", a[c*k+d]);
			}
			printf("\n");
		}
		printf("\n");
	}*/

	oclWrite(_args[0].p, 0, sizeof(float)*mk, a);
	oclWrite(_args[0].p, sizeof(float)*mk, sizeof(float)*kn, b);
	if (beta!=0) oclWrite(_args[0].p, sizeof(float)*(mk+kn), sizeof(float)*mn, c);

#if 1
	if (ta=='T') {
		_info[0] = m;	// a
		_info[1] = k;	// ta
		_info[2] = 0;	// input a
		_info[3] = off_a = mk +kn +mn;
		KTRANSPOSE.global_size[0] = ceil_int(m, TRANSPOSEX);
		KTRANSPOSE.global_size[1] = ceil_int(k, TRANSPOSEY);

/*		_info[0] = k;	// a
		_info[1] = m;	// ta
		_info[2] = 0;	// input a
		_info[3] = off_a = mk +kn +mn;
		KTRANSPOSE.global_size[0] = ceil_int(k, TRANSPOSEX);
		KTRANSPOSE.global_size[1] = ceil_int(m, TRANSPOSEY);*/

		oclKernelArgsWrite(_args);
		oclRun(&KTRANSPOSE);
	}
#endif
	if (tb=='T') {
		_info[0] = k;	// b
		_info[1] = n;	// tb
		_info[2] = mk;	// input b
		_info[3] = off_b = mk +kn +mn +mk;
		KTRANSPOSE.global_size[0] = ceil_int(k, TRANSPOSEX);
		KTRANSPOSE.global_size[1] = ceil_int(n, TRANSPOSEY);

		oclKernelArgsWrite(_args);
		oclRun(&KTRANSPOSE);
	}

	_info[0] = m;
	_info[1] = n;
	_info[2] = k;
	_info[3] = off_a;	// a
	_info[4] = off_b;	// b
	_info[5] = mk +kn;	// c
	_param[0] = alpha;
	_param[1] = beta;
	KGEMM_RNN.global_size[0] = ceil_int(m, TS);
	KGEMM_RNN.global_size[1] = ceil_int(n, TS);

	oclKernelArgsWrite(_args);
	oclRun(&KGEMM_RNN);
	oclRead(_args[0].p, sizeof(float)*(mk+kn), sizeof(float)*mn, c);
}
void sgemm_ocl_finish()
{
	oclReleaseKernel(_kernel, _ksz);
	oclFinish();
}
static inline void ocl_im2col(float *inputs, int ich, int w, int h, int k, int pad, int stride, float *outputs)
{
	// im2col(pix, 3, h, w, 4, 4, 2, 2, 1, 1, workspace);
	int hcol = (h + 2 * pad - k) / stride + 1;
	int wcol = (w + 2 * pad - k) / stride + 1;
	_info[0] = wcol*hcol*ich*k*k;	// inputs
	_info[1] = ich;
	_info[2] = h;
	_info[3] = w;
	_info[4] = k;
	_info[5] = pad;
	_info[6] = stride;
	_info[7] = 0;			// outputs
	KIM2COL.global_size[0] = ceil_int(_info[0], 16);
	oclWrite(_args[0].p, sizeof(float)*_info[0], sizeof(float)*w*h*ich, inputs);
	oclRun(&KIM2COL);
	oclRead(_args[0].p, sizeof(float)*_info[7], sizeof(float)*_info[0], outputs);
}
static inline void ocl_convolution(float *inputs, int ich, int w, int h, float *weights, int k, int pad, int stride, float *outputs, int ch)
{
	// im2col(pix, 3, h, w, 4, 4, 2, 2, 1, 1, workspace);
	int hcol = (h + 2 * pad - k) / stride + 1;
	int wcol = (w + 2 * pad - k) / stride + 1;
	oclWrite(_args[0].p, sizeof(float)*wcol*hcol*ich*k*k, sizeof(float)*w*h*ich, inputs);
	_info[0] = wcol*hcol*ich*k*k;	// inputs
	_info[1] = ich;
	_info[2] = h;
	_info[3] = w;
	_info[4] = k;
	_info[5] = pad;
	_info[6] = stride;
	_info[7] = 0;			// outputs
	KIM2COL.global_size[0] = ceil_int(_info[0], 16);
	oclRun(&KIM2COL);

	// sgemm_ocl('N', 'T', ch, wcol*hcol, k*k, magic_kernel, workspace, pix);
	oclWrite(_args[0].p, sizeof(float)*(wcol*hcol*ich*k*k), sizeof(float)*k*k*ich*ch, weights);
	_info[0] = ch;
	_info[1] = wcol*hcol /* *batch */;
	_info[2] = k*k*ich;
	_info[3] = wcol*hcol*ich*k*k;			// a (weights)
	_info[4] = 0;					// b (col)
	_info[5] = wcol*hcol*ich*k*k +k*k*ich*ch;	// c
	KGEMM_RNN.global_size[0] = ceil_int(_info[0], TS);
	KGEMM_RNN.global_size[1] = ceil_int(_info[1], TS);
	oclRun(&KGEMM_RNN);
	oclRead(_args[0].p, sizeof(float)*_info[5], sizeof(float)*wcol*hcol*ch, outputs);
}
#if 0
static inline void ocl_convolution_LReLU(float *inputs, int ich, int w, int h, float *weights, int k, int pad, int stride, float *outputs, int ch, float *bias)
{
	// im2col(pix, 3, h, w, 4, 4, 2, 2, 1, 1, workspace);
	int hcol = (h + 2 * pad - k) / stride + 1;
	int wcol = (w + 2 * pad - k) / stride + 1;
	_info[0] = wcol*hcol*ich*k*k;	// inputs
	_info[1] = ich;
	_info[2] = h;
	_info[3] = w;
	_info[4] = k;
	_info[5] = pad;
	_info[6] = stride;
	_info[7] = 0;			// outputs
	_kernel[2].global_size[0] = ceil_int(_info[0], 16);
//	printf("clEnqueueWriteBuffer: %lu %lu\n", sizeof(float)*_info[0], sizeof(float)*w*h*ich);
	oclWrite(_args[0].p, sizeof(float)*_info[0], sizeof(float)*w*h*ich, inputs);
	oclRun(_kernel+2);

	// sgemm_ocl('N', 'T', ch, wcol*hcol, k*k, magic_kernel, workspace, pix);
	_info[0] = ch;
	_info[1] = wcol*hcol /* *batch */;
	_info[2] = k*k*ich;
	_info[3] = wcol*hcol*ich*k*k;			// a (weights)
	_info[4] = 0;					// b (col)
	_info[5] = wcol*hcol*ich*k*k +k*k*ich*ch;	// c
	_info[6] = _info[5] + wcol*hcol*ch;
	_kernel[3].global_size[0] = ceil_int(_info[0], TS);
	_kernel[3].global_size[1] = ceil_int(_info[1], TS);
//	printf("clEnqueueWriteBuffer: %lu %lu\n", sizeof(float)*_info[3], sizeof(float)*k*k*ich*ch);
	oclWrite(_args[0].p, sizeof(float)*_info[3], sizeof(float)*k*k*ich*ch, weights);
//	printf("clEnqueueWriteBuffer: %lu %lu\n", sizeof(float)*_info[6], sizeof(float)*ch);
	oclWrite(_args[0].p, sizeof(float)*_info[6], sizeof(float)*ch, bias);
	oclRun(_kernel+3);
//	printf("clEnqueueReadBuffer: %lu %lu\n", sizeof(float)*_info[5], sizeof(float)*wcol*hcol*ch);
	oclRead(_args[0].p, sizeof(float)*_info[5], sizeof(float)*wcol*hcol*ch, outputs);
}
#endif
