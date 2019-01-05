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

/*[----------] Printing best result in database format to stdout
{ "AMD Radeon HD 7800 Series (TAHITI / DRM 3.23.0 / 4.16.11-berry, LLVM 6.0.1)", { {"MWG",64}, {"NWG",64}, {"KWG",32}, {"MDIMC",8}, {"NDIMC",8}, {"MDIMA",32}, {"NDIMB",8}, {"KWI",2}, {"VWM",2}, {"VWN",8}, {"STRM",1}, {"STRN",0}, {"SA",0}, {"SB",1}, {"PRECISION",32} } }
[ -------> ] 12.5 ms or 1374.484 GFLOPS*/
#define MWG	64
#define NWG	64
#define KWG	32
#define MDIMC	8
#define NDIMC	8
#define MDIMA	32
#define NDIMB	8
#define KWI	2
#define VWM	2
#define VWN	8
#define STRM	1
#define STRN	0
#define SA	0
#define SB	1
#define PRECISION	32

// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains an example OpenCL kernel as part of the gemm.cc example. It is an optimized
// matrix-multiplication kernel according to the paper by Matsumoto et al. and the tutorial on
// http://www.cedricnugteren.nl/tutorial.php. It is fully configurable (and tunable!) using more or
// less the same parameters/naming conventions as in the paper. It supports single and double
// precision (SGEMM/DGEMM) through a pre-processor define.
//
// Note: this kernel requires a compiler compliant to OpenCL 1.1 or higher.
//
// -------------------------------------------------------------------------------------------------
//
// Copyright 2014 SURFsara
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//  http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// =================================================================================================
//
// Matrices are accessed as follows:
// A: [k*M + m], with 'k' ranging from 0:K and 'm' from 0:M (m,k,m)
// B: [k*N + n], with 'k' ranging from 0:K and 'n' from 0:N (n,k,n)
// C: [n*M + m], with 'n' ranging from 0:N and 'm' from 0:M (m,n,m)
//
// Or as an image (assuming column-major)
//       K                      
//    o-------o                 
//    |       |                 
//  N | [B^T] |                 
//    |       |                 
//    o-------o                 
//        K               N     
//    o-------o        o-----o  
//  M |  [A]  |      M | [C] |  
//    |       |        |     |  
//    o-------o        o-----o  
//                              
//
// Parameters determined by the tuner
// MWG       : Tile-size in dimension M (e.g. 64, 128)
// NWG       : Tile-size in dimension N (e.g. 64, 128)
// KWG       : Tile-size in dimension K (e.g. 8, 16)
// MDIMC     : Threads per workgroup in M-dimension (e.g. 8, 16, 32)
// NDIMC     : Threads per workgroup in N-dimension (e.g. 8, 16, 32)
// MDIMA     : Re-shaped tile dimension of matrix A: KDIMA * MDIMA
// NDIMB     : Re-shaped tile dimension of matrix B: KDIMB * NDIMB
// KWI       : Unroll factor of the KWG loop (smaller or equal than KWG)
// VWM       : Vector width of matrices A and C (supported 1, 2, 4, and 8)
// VWN       : Vector width of matrix B (supported 1, 2, 4, and 8)
// STRM      : Use strided access within a thread in the M-dimension (1) or not (0)
// STRN      : Use strided access within a thread in the N-dimension (1) or not (0)
// SA        : Use local/shared memory to cache matrix A (1) or not (0)
// SB        : Use local/shared memory to cache matrix B (1) or not (0)
// PRECISION : Whether to use single (32) or double (64) precision data-types
//
// =================================================================================================

// Helper parameters based on the above tuning parameters
#define MWI (MWG/MDIMC)               // Work per work-item (M-dimension)
#define NWI (NWG/NDIMC)               // Work per work-item (N-dimension)
#define KDIMA ((MDIMC*NDIMC)/(MDIMA)) // Re-shaped tile dimension of matrix A: KDIMA * MDIMA
#define KDIMB ((MDIMC*NDIMC)/(NDIMB)) // Re-shaped tile dimension of matrix B: KDIMB * NDIMB
#define MWA (MWG/MDIMA)               // Amount of loads-per-thread for matrix A (M-dimension)
#define KWA (KWG/KDIMA)               // Amount of loads-per-thread for matrix A (K-dimension)
#define KWB (KWG/KDIMB)               // Amount of loads-per-thread for matrix B (K-dimension)
#define NWB (NWG/NDIMB)               // Amount of loads-per-thread for matrix B (N-dimension)

// Settings
#define USE_VECTOR_MAD 1              // Don't unroll the vector MAD computation
#define USE_CL_MAD 0                  // Uses the non-IEEE754 compliant OpenCL mad() (if above is 0)

// =================================================================================================

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

// =================================================================================================

// Data-widths in dimension M
#if VWM == 1
    typedef real realM;
#elif VWM == 2
    typedef real2 realM;
#elif VWM == 4
    typedef real4 realM;
#elif VWM == 8
    typedef real8 realM;
#elif VWM == 16
    typedef real16 realM;
#endif

// Data-widths in dimension N
#if VWN == 1
    typedef real realN;
#elif VWN == 2
    typedef real2 realN;
#elif VWN == 4
    typedef real4 realN;
#elif VWN == 8
    typedef real8 realN;
#elif VWN == 16
    typedef real16 realN;
#endif

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
#if SA == 1
inline void GlobalToLocalA(const __global realM* restrict agm, __local realM* alm,
                           const int kSizeM, const int tid, const int kwg) {
  const int la0 = tid % MDIMA;
  const int la1 = tid / MDIMA;
  \n\x23pragma unroll\n
  for (int mia=0; mia<MWA/VWM; ++mia) {
    \n\x23pragma unroll\n
    for (int kia=0; kia<KWA; ++kia) {

      // Computes the indices based on strided/non-strided access
      #if STRM == 0
        int mg = mia + la0*(MWA/VWM);
      #elif STRM == 1
        int mg = la0 + mia*MDIMA;
      #endif

      // Computes the indices for the global memory
      int kg = kia + la1*KWA;
      int idm = mg + get_group_id(0)*(MWG/VWM);
      int idk = kg + kwg;

      // Loads the data from global memory (not transposed) into the local memory
      alm[kg*(MWG/VWM) + mg] = agm[idk*(kSizeM/VWM) + idm];
    }
  }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
inline void GlobalToLocalB(const __global realN* restrict bgm, __local realN* blm,
                           const int kSizeN, const int tid, const int kwg) {
  const int lb0 = tid % NDIMB;
  const int lb1 = tid / NDIMB;
  \n\x23pragma unroll\n
  for (int kib=0; kib<KWB; ++kib) {
    \n\x23pragma unroll\n
    for (int nib=0; nib<NWB/VWN; ++nib) {

      // Computes the indices based on strided/non-strided access
      #if STRN == 0
        int ng = nib + lb0*(NWB/VWN);
      #elif STRN == 1
        int ng = lb0 + nib*NDIMB;
      #endif

      // Computes the indices for the global memory
      int kg = kib + lb1*KWB;
      int idn = ng + get_group_id(1)*(NWG/VWN);
      int idk = kg + kwg;

      // Loads the data from global memory (transposed) into the local memory
      blm[kg*(NWG/VWN) + ng] = bgm[idk*(kSizeN/VWN) + idn];
    }
  }
}
#endif

// =================================================================================================

// Caches global off-chip memory directly into per-thread private memory (registers). This function
// is specific for caching the A input matrix.
#if SA == 0
inline void GlobalToPrivateA(const __global realM* restrict agm, realM apm[MWI/VWM],
                             const int kSizeM, const int idk, const int kwg) {
  \n\x23pragma unroll\n
  for (int mi=0; mi<MWI/VWM; ++mi) {

    // Computes the indices based on strided/non-strided access
    #if STRM == 0
      int mg = mi + get_local_id(0)*(MWI/VWM);
    #elif STRM == 1
      int mg = get_local_id(0) + mi*MDIMC;
    #endif

    // Computes the indices for the global memory
    int idm = mg + get_group_id(0)*(MWG/VWM);

    // Loads the data from global memory (not transposed) and stores into registers
    apm[mi] = agm[idk*(kSizeM/VWM) + idm];
  }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 0
inline void GlobalToPrivateB(const __global realN* restrict bgm, realN bpm[NWI/VWN],
                             const int kSizeN, const int idk) {
  \n\x23pragma unroll\n
  for (int ni=0; ni<NWI/VWN; ++ni) {

    // Computes the indices based on strided/non-strided access
    #if STRN == 0
      int ng = ni + get_local_id(1)*(NWI/VWN);
    #elif STRN == 1
      int ng = get_local_id(1) + ni*NDIMC;
    #endif

    // Computes the indices for the global memory
    int idn = ng + get_group_id(1)*(NWG/VWN);

    // Loads the data from global memory (transposed) and stores into registers
    bpm[ni] = bgm[idk*(kSizeN/VWN) + idn];
  }
}
#endif

// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
#if SA == 1
inline void LocalToPrivateA(__local realM* alm, realM apm[MWI/VWM], const int kg) {
  \n\x23pragma unroll\n
  for (int mi=0; mi<MWI/VWM; ++mi) {
    #if STRM == 0
      int mg = mi + get_local_id(0)*(MWI/VWM);
    #elif STRM == 1
      int mg = get_local_id(0) + mi*MDIMC;
    #endif
    apm[mi] = alm[kg*(MWG/VWM) + mg];
  }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
inline void LocalToPrivateB(__local realN* blm, realN bpm[NWI/VWN], const int kg) {
  \n\x23pragma unroll\n
  for (int ni=0; ni<NWI/VWN; ++ni) {
    #if STRN == 0
      int ng = ni + get_local_id(1)*(NWI/VWN);
    #elif STRN == 1
      int ng = get_local_id(1) + ni*NDIMC;
    #endif
    bpm[ni] = blm[kg*(NWG/VWN) + ng];
  }
}
#endif

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm
inline void StoreResults(__global realM* cgm, realM cpm[NWI][MWI/VWM], const int kSizeM) {
  \n\x23pragma unroll\n
  for (int ni=0; ni<NWI; ++ni) {
    \n\x23pragma unroll\n
    for (int mi=0; mi<MWI/VWM; ++mi) {
      #if STRM == 0
        int mg = mi + get_local_id(0)*(MWI/VWM);
      #elif STRM == 1
        int mg = get_local_id(0) + mi*MDIMC;
      #endif
      #if STRN == 0
        int ng = ni + get_local_id(1)*NWI;
      #elif STRN == 1
        int ng = ni%VWN + get_local_id(1)*VWN + (ni/VWN)*VWN*NDIMC;
      #endif
      int idm = mg + get_group_id(0)*(MWG/VWM);
      int idn = ng + get_group_id(1)*NWG;
      int index = idn*(kSizeM/VWM) + idm;
      cgm[index] = cpm[ni][mi];
    }
  }
}

// =================================================================================================

// The basic scalar multiply-add function
#if USE_CL_MAD == 1
  #define MultiplyAdd(cval, aval, bval) (cval = mad(aval, bval, cval))
#else
  #define MultiplyAdd(cval, aval, bval) (cval += (aval) * (bval))
#endif

// The vectorised multiply-add function
inline realM MultiplyAddVector(realM cvec, const realM avec, const real bval) {
  #if USE_VECTOR_MAD == 1
    cvec += avec * bval;
  #else
    #if VWM == 1
      MultiplyAdd(cvec,    avec,    bval);
    #elif VWM == 2
      MultiplyAdd(cvec.x , avec.x,  bval);
      MultiplyAdd(cvec.y , avec.y,  bval);
    #elif VWM == 4
      MultiplyAdd(cvec.x , avec.x,  bval);
      MultiplyAdd(cvec.y , avec.y,  bval);
      MultiplyAdd(cvec.z , avec.z,  bval);
      MultiplyAdd(cvec.w , avec.w,  bval);
    #elif VWM == 8
      MultiplyAdd(cvec.s0, avec.s0, bval);
      MultiplyAdd(cvec.s1, avec.s1, bval);
      MultiplyAdd(cvec.s2, avec.s2, bval);
      MultiplyAdd(cvec.s3, avec.s3, bval);
      MultiplyAdd(cvec.s4, avec.s4, bval);
      MultiplyAdd(cvec.s5, avec.s5, bval);
      MultiplyAdd(cvec.s6, avec.s6, bval);
      MultiplyAdd(cvec.s7, avec.s7, bval);
    #elif VWM == 16
      MultiplyAdd(cvec.s0, avec.s0, bval);
      MultiplyAdd(cvec.s1, avec.s1, bval);
      MultiplyAdd(cvec.s2, avec.s2, bval);
      MultiplyAdd(cvec.s3, avec.s3, bval);
      MultiplyAdd(cvec.s4, avec.s4, bval);
      MultiplyAdd(cvec.s5, avec.s5, bval);
      MultiplyAdd(cvec.s6, avec.s6, bval);
      MultiplyAdd(cvec.s7, avec.s7, bval);
      MultiplyAdd(cvec.s8, avec.s8, bval);
      MultiplyAdd(cvec.s9, avec.s9, bval);
      MultiplyAdd(cvec.sA, avec.sA, bval);
      MultiplyAdd(cvec.sB, avec.sB, bval);
      MultiplyAdd(cvec.sC, avec.sC, bval);
      MultiplyAdd(cvec.sD, avec.sD, bval);
      MultiplyAdd(cvec.sE, avec.sE, bval);
      MultiplyAdd(cvec.sF, avec.sF, bval);
    #endif
  #endif
  return cvec;
}

// Performs the actual computation: Cpm += Apm * Bpm
inline void MultiplyAccumulate(realM cpm[NWI][MWI/VWM], realM apm[MWI/VWM], realN bpm[NWI/VWN]) {
  \n\x23pragma unroll\n
  for (int ni=0; ni<NWI/VWN; ++ni) {
    \n\x23pragma unroll\n
    for (int mi=0; mi<MWI/VWM; ++mi) {
      #if VWN == 1
        cpm[ni*VWN + 0][mi] = MultiplyAddVector(cpm[ni*VWN + 0][mi], apm[mi], bpm[ni]);
      #elif VWN == 2
        cpm[ni*VWN + 0][mi] = MultiplyAddVector(cpm[ni*VWN + 0][mi], apm[mi], bpm[ni].x);
        cpm[ni*VWN + 1][mi] = MultiplyAddVector(cpm[ni*VWN + 1][mi], apm[mi], bpm[ni].y);
      #elif VWN == 4
        cpm[ni*VWN + 0][mi] = MultiplyAddVector(cpm[ni*VWN + 0][mi], apm[mi], bpm[ni].x);
        cpm[ni*VWN + 1][mi] = MultiplyAddVector(cpm[ni*VWN + 1][mi], apm[mi], bpm[ni].y);
        cpm[ni*VWN + 2][mi] = MultiplyAddVector(cpm[ni*VWN + 2][mi], apm[mi], bpm[ni].z);
        cpm[ni*VWN + 3][mi] = MultiplyAddVector(cpm[ni*VWN + 3][mi], apm[mi], bpm[ni].w);
      #elif VWN == 8
        cpm[ni*VWN + 0][mi] = MultiplyAddVector(cpm[ni*VWN + 0][mi], apm[mi], bpm[ni].s0);
        cpm[ni*VWN + 1][mi] = MultiplyAddVector(cpm[ni*VWN + 1][mi], apm[mi], bpm[ni].s1);
        cpm[ni*VWN + 2][mi] = MultiplyAddVector(cpm[ni*VWN + 2][mi], apm[mi], bpm[ni].s2);
        cpm[ni*VWN + 3][mi] = MultiplyAddVector(cpm[ni*VWN + 3][mi], apm[mi], bpm[ni].s3);
        cpm[ni*VWN + 4][mi] = MultiplyAddVector(cpm[ni*VWN + 4][mi], apm[mi], bpm[ni].s4);
        cpm[ni*VWN + 5][mi] = MultiplyAddVector(cpm[ni*VWN + 5][mi], apm[mi], bpm[ni].s5);
        cpm[ni*VWN + 6][mi] = MultiplyAddVector(cpm[ni*VWN + 6][mi], apm[mi], bpm[ni].s6);
        cpm[ni*VWN + 7][mi] = MultiplyAddVector(cpm[ni*VWN + 7][mi], apm[mi], bpm[ni].s7);
      #elif VWN == 16
        cpm[ni*VWN + 0 ][mi] = MultiplyAddVector(cpm[ni*VWN + 0 ][mi], apm[mi], bpm[ni].s0);
        cpm[ni*VWN + 1 ][mi] = MultiplyAddVector(cpm[ni*VWN + 1 ][mi], apm[mi], bpm[ni].s1);
        cpm[ni*VWN + 2 ][mi] = MultiplyAddVector(cpm[ni*VWN + 2 ][mi], apm[mi], bpm[ni].s2);
        cpm[ni*VWN + 3 ][mi] = MultiplyAddVector(cpm[ni*VWN + 3 ][mi], apm[mi], bpm[ni].s3);
        cpm[ni*VWN + 4 ][mi] = MultiplyAddVector(cpm[ni*VWN + 4 ][mi], apm[mi], bpm[ni].s4);
        cpm[ni*VWN + 5 ][mi] = MultiplyAddVector(cpm[ni*VWN + 5 ][mi], apm[mi], bpm[ni].s5);
        cpm[ni*VWN + 6 ][mi] = MultiplyAddVector(cpm[ni*VWN + 6 ][mi], apm[mi], bpm[ni].s6);
        cpm[ni*VWN + 7 ][mi] = MultiplyAddVector(cpm[ni*VWN + 7 ][mi], apm[mi], bpm[ni].s7);
        cpm[ni*VWN + 8 ][mi] = MultiplyAddVector(cpm[ni*VWN + 8 ][mi], apm[mi], bpm[ni].s8);
        cpm[ni*VWN + 9 ][mi] = MultiplyAddVector(cpm[ni*VWN + 9 ][mi], apm[mi], bpm[ni].s9);
        cpm[ni*VWN + 10][mi] = MultiplyAddVector(cpm[ni*VWN + 10][mi], apm[mi], bpm[ni].sA);
        cpm[ni*VWN + 11][mi] = MultiplyAddVector(cpm[ni*VWN + 11][mi], apm[mi], bpm[ni].sB);
        cpm[ni*VWN + 12][mi] = MultiplyAddVector(cpm[ni*VWN + 12][mi], apm[mi], bpm[ni].sC);
        cpm[ni*VWN + 13][mi] = MultiplyAddVector(cpm[ni*VWN + 13][mi], apm[mi], bpm[ni].sD);
        cpm[ni*VWN + 14][mi] = MultiplyAddVector(cpm[ni*VWN + 14][mi], apm[mi], bpm[ni].sE);
        cpm[ni*VWN + 15][mi] = MultiplyAddVector(cpm[ni*VWN + 15][mi], apm[mi], bpm[ni].sF);
      #endif
    }
  }
}

// =================================================================================================

// Main entry of the kernel. This function contains the basic skeleton, the functionality is
// provided by the inlined functions above.
__attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
__kernel void gemm_fast(const int kSizeM, const int kSizeN, const int kSizeK,
                        const __global realM* restrict agm,
                        const __global realN* restrict bgm,
                        __global realM* cgm) {

  // Combined thread identifier
  #if SA == 1 || SB == 1
    volatile int tid = get_local_id(0) + MDIMC*get_local_id(1);
  #endif

  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realN blm[KWG * NWG/VWN];
  #endif
  
  // Allocates workitem-private memory (registers)
  realM apm[MWI/VWM];
  realN bpm[NWI/VWN];
  realM cpm[NWI][MWI/VWM];

  // Initializes the accumulation registers
  \n\x23pragma unroll\n
  for (int mi=0; mi<MWI/VWM; ++mi) {
    \n\x23pragma unroll\n
    for (int ni=0; ni<NWI; ++ni) {
      cpm[ni][mi] = (realM)ZERO;
    }
  }

  // Loops over all workgroup tiles
  for (int kwg=0; kwg<kSizeK; kwg+=KWG) {

    // Loads data: off-chip --> local (matrix A)
    #if SA == 1
      GlobalToLocalA(agm, alm, kSizeM, tid, kwg);
    #endif
    // Loads data: off-chip --> local (matrix B)
    #if SB == 1
      GlobalToLocalB(bgm, blm, kSizeN, tid, kwg);
    #endif

    // Synchronizes all threads in a workgroup
    #if SA == 1 || SB == 1
      barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    // Loops over all workitem tiles, unrolled by a factor KWI
    for (int pwi=0; pwi<KWG; pwi+=KWI) {
      \n\x23pragma unroll\n
      for (int pit=0; pit<KWI; ++pit) {
        #if SA == 0 || SB == 0
          int idk = kwg + pwi + pit;
        #endif
        #if SA == 1 || SB == 1
          int kg = pwi+pit;
        #endif

        // Loads data: local --> private (matrix A)
        #if SA == 1
          LocalToPrivateA(alm, apm, kg);
        // Loads data: off-chip --> private (matrix A)
        #else
          GlobalToPrivateA(agm, apm, kSizeM, idk, kwg);
        #endif

        // Loads data: local --> private (matrix B)
        #if SB == 1
          LocalToPrivateB(blm, bpm, kg);
        // Loads data: off-chip --> private (matrix B)
        #else
          GlobalToPrivateB(bgm, bpm, kSizeN, idk);
        #endif

        // Performs the accumulation (Cpm += Apm * Bpm)
        MultiplyAccumulate(cpm, apm, bpm);
      }
    }

    // Synchronizes all threads in a workgroup
    #if SA == 1 || SB == 1
      barrier(CLK_LOCAL_MEM_FENCE);
    #endif
  }

  // Stores an MWG * NWG tile of results
  StoreResults(cgm, cpm, kSizeM);
}

// =================================================================================================

);

// Size of the matrices - K, M, N (squared)
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
ocl_t kernel[] = { // *8/64
	{ "gemm_fast", 0, 2,{/*M*/SIZE*MDIMC/MWG,/*N*/SIZE*NDIMC/NWG},{1*MDIMC,1*NDIMC}, args },
//	{ "gemm_fast", 0, 2,{SIZE,SIZE},{1,1}, args },
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
