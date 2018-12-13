/*! \file gemm.h
    \brief none-assembly version of GEMM algorithm for FIXED-POINT datatype
*/
#ifndef GEMM_H
#define GEMM_H
#include "setting.h"
/// packed height of A
#define MC  512
/// packed width of A, height of B
#define KC  1024
/// packed width of B
#define NC  2048
/// width of micro kernel
#define MR  4
/// height of micro kernel
#define NR  4

/// storage for packed A
static Dtype _A[MC*KC];
/// storage for packed B
static Dtype _B[KC*NC];

/*
 *  Packing a micro panel from A, i.e. micro panels are minimum size with width being 4
 */
static void
pack_MRxk(int k, const Dtype *A, int incRowA, int incColA,
          Dtype *buffer)
{
    int i, j;

    for (j=0; j<k; ++j) {
        for (i=0; i<MR; ++i) {
            buffer[i] = A[i*incRowA];
        }
        buffer += MR;
        A      += incColA;
    }
}

/*
 *  Packing panels from A
 */
static void
pack_A(int mc, int kc, const Dtype *A, int incRowA, int incColA,
       Dtype *buffer)
{
    int mp  = mc / MR;
    int _mr = mc % MR;

    int i, j;

    for (i=0; i<mp; ++i) {
        pack_MRxk(kc, A, incRowA, incColA, buffer);
        buffer += kc*MR;
        A      += MR*incRowA;
    }
    // needs to work with unit of 4, if not, pack with zero
    if (_mr>0) {
        for (j=0; j<kc; ++j) {
            for (i=0; i<_mr; ++i) {
                buffer[i] = A[i*incRowA];
            }
            for (i=_mr; i<MR; ++i) {
                buffer[i] = 0.0;
            }
            buffer += MR;
            A      += incColA;
        }
    }
}

/*
 *  Packing a complete micro panels from B
 */
static void
pack_kxNR(int k, const Dtype *B, int incRowB, int incColB,
          Dtype *buffer)
{
    int i, j;

    for (i=0; i<k; ++i) {
        for (j=0; j<NR; ++j) {
            buffer[j] = B[j*incColB];
        }
        buffer += NR;
        B      += incRowB;
    }
}

/*
 *  this guarantees within the micro kernel the consecutive memory access are to contiguous cache 
 */
static void
pack_B(int kc, int nc, const Dtype *B, int incRowB, int incColB,
       Dtype *buffer)
{
    int np  = nc / NR;
    int _nr = nc % NR;

    int i, j;

    for (j=0; j<np; ++j) {
        pack_kxNR(kc, B, incRowB, incColB, buffer);
        buffer += kc*NR;
        B      += NR*incColB;
    }
    // needs to work in unit of 4, if not, pack with zero
    if (_nr>0) {
        for (i=0; i<kc; ++i) {
            for (j=0; j<_nr; ++j) {
                buffer[j] = B[j*incColB];
            }
            for (j=_nr; j<NR; ++j) {
                buffer[j] = 0.0;
            }
            buffer += NR;
            B      += incRowB;
        }
    }
}

/*
 * Micro kernel for multiplying panels from A and B.
 * it performs kc*16 add_multiply operations and 16 store
 * an assembly optimized version shall have 16 registers to accumulate the values
 */
static void
dgemm_micro_kernel(int kc,
                   const Dtype *A, const Dtype *B,
                   Dtype *C, int incRowC, int incColC)
{

    // temporary values to hold accumulated sum, avoid shift operations
    Dtype AB[MR*NR];
    
    int i, j, l;

    for (l=0; l<MR*NR; ++l) {
        AB[l] = 0;
    }

    for (l=0; l<kc; ++l) {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] += add_multiply(AB[i+j*MR], A[i], B[j]);
            }
        }
        A += MR;
        B += NR;
    }
}

/*
 *  Macro Kernel for the multiplication of blocks of A and B.  We assume that
 *  these blocks were previously packed to buffers _A and _B.
 */
static void
dgemm_macro_kernel(int     mc,
                   int     nc,
                   int     kc,
                   Dtype  *C,
                   int     incRowC,
                   int     incColC)
{
    int mp = (mc+MR-1) / MR;
    int np = (nc+NR-1) / NR;

    int mr, nr;
    int i, j;

    for (j=0; j<np; ++j) {
        for (i=0; i<mp; ++i) {
            dgemm_micro_kernel(kc, 
                               &_A[i*kc*MR], &_B[j*kc*NR],
                               &C[i*MR*incRowC+j*NR*incColC],
                               incRowC, incColC);
        }
    }
}

/// this functions computes C += A*B
/// note it's not C = A*B, it is assumed C is initialized to some desired values
/// hence, in convolution layer, C is first initialized with the bias value
/// and then A*B is added to each pixel
void
dgemm_nn(int            m,
         int            n,
         int            k,
         const Dtype   *A,
         int            incRowA,
         const Dtype   *B,
         int            incRowB,
         Dtype         *C,
         int            incRowC
         )
{

    int mb = (m+MC-1) / MC;
    int nb = (n+NC-1) / NC;
    int kb = (k+KC-1) / KC;

    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;

    int mc, nc, kc;
    int i, j, l;

    for (j=0; j<nb; ++j) {
        nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        for (l=0; l<kb; ++l) {
            kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;

            pack_B(kc, nc,
                   &B[l*KC*incRowB+j*NC], incRowB, 1,
                   _B);

            for (i=0; i<mb; ++i) {
                mc = (i!=mb-1 || _mc==0) ? MC : _mc;

                pack_A(mc, kc,
                       &A[i*MC*incRowA+l*KC], incRowA, 1,
                       _A);

                dgemm_macro_kernel(mc, nc, kc,
                                   &C[i*MC*incRowC+j*NC],
                                   incRowC, 1);
            }
        }
    }
}
#endif