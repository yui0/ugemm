/* public domain Simple, Minimalistic, Fast GEMM library
 *	©2018 Yuichiro Nakada
 *
 * Basic usage:
 *
 * */

#include <memory.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#define malloc_a(size, alignment)	_aligned_malloc(size, alignment)
#define free_a(p)			_aligned_free(p)
#else
#define malloc_a(size, alignment)	({ void* p; posix_memalign((void**) &p, alignment, size) == 0 ? p : NULL; })
#define free_a(p)			free(p)
#endif  /* _MSC_VER */

//#include "dgemm_sse.h"
#include "dgemm_avx.h"
#include "sgemm_avx256.h"

#define real		double
#define GEMM(def)	_dgemm##def
#include "gemm_cpu.h"
#undef GEMM
#undef real

#define real		float
#define GEMM(def)	sgemm##def
#include "gemm_cpu.h"
#undef GEMM
#undef real

void daxpy_avx(int N, double a, double* const restrict x, double* const restrict y, double* restrict z)
{
	int i;
	const __m256d va = _mm256_broadcast_sd(&a);
	#pragma omp parallel for
	for (i=0; i<N; i+=4) {
		__m256d vx = _mm256_load_pd(x + i);
		__m256d vy = _mm256_load_pd(y + i);
#ifdef __FMA__
		__m256d vz = _mm256_fmadd_pd(va, vx, vy);
#else
		__m256d vz = _mm256_add_pd(va, _mm256_mul_pd(vx, vy));
#endif
		_mm256_stream_pd(z + i, vz);
	}
}
void saxpy_avx(int N, float a, float* const restrict x, float* const restrict y, float* restrict z)
{
	int i;
	const __m256 va = _mm256_broadcast_ss(&a);
	#pragma omp parallel for
	for (i=0; i<N; i+=8) {
		__m256 vx = _mm256_load_ps(x + i);
		__m256 vy = _mm256_load_ps(y + i);
#ifdef __FMA__
		__m256 vz = _mm256_fmadd_ps(va, vx, vy);
#else
		__m256 vz = _mm256_add_ps(va, _mm256_mul_ps(vx, vy));
#endif
		_mm256_stream_ps(z + i, vz);
	}
}

void daxpy_cpu(
	const int	N,
	const double	alpha,
	const double	*x,
	const int	incx,
	double		*y,
	const int	incy)
{
	for (int n=0; n<N; n++) {
		y[n * incy] += alpha * x[n * incx];
	}
}
void saxpy_cpu(
	const int	N,
	const float	alpha,
	const float	*x,
	const int	incx,
	float		*y,
	const int	incy)
{
	for (int n=0; n<N; n++) {
		y[n * incy] += alpha * x[n * incx];
	}
}

void dgemv_cpu(
	char		trans,
	const int	M,
	const int	N,
	const double	alpha,
	const double	*A,
	const int	lda,
	const double	*x,
	const int	incx,
	const double	beta,
	double		*y,
	const int	incy)
{
	if (trans == 'N') {
		for (int m=0; m<M; m++) {
			register double sum = 0.0;
			for (int n=0; n<N; n++) {
				sum += A[m + n * lda] * x[n * incy];
			}
			y[m * incy] = alpha * sum + beta * y[m * incy];
		}
	} else /*if (trans == 'T')*/ {
		for (int m=0; m<M; m++) {
			register double sum = 0.0;
			for (int n=0; n<N; n++) {
				sum += A[n + m * lda] * x[n * incy];
			}
			y[m * incy] = alpha * sum + beta * y[m * incy];
		}
	}
}
void sgemv_cpu(
	char		trans,
	const int	M,
	const int	N,
	const float	alpha,
	const float	*A,
	const int	lda,
	const float	*x,
	const int	incx,
	const float	beta,
	float		*y,
	const int	incy)
{
	if (trans == 'N') {
		for (int m=0; m<M; m++) {
			register float sum = 0.0;
			for (int n=0; n<N; n++) {
				sum += A[m + n * lda] * x[n * incy];
			}
			y[m * incy] = alpha * sum + beta * y[m * incy];
		}
	} else /*if (trans == 'T')*/ {
		for (int m=0; m<M; m++) {
			register float sum = 0.0;
			for (int n=0; n<N; n++) {
				sum += A[n + m * lda] * x[n * incy];
			}
			y[m * incy] = alpha * sum + beta * y[m * incy];
		}
	}
}

void dgemm_cpu(
	char		major,
	char		transa,
	char		transb,
	const int	M,
	const int	N,
	const int	K,
	const double	alpha,
	const double	*A,
	const int	lda,
	const double	*B,
	const int	ldb,
	const double	beta,
	double		*C,
	const int	ldc)
{
	// RowMajor
	if (major == 'R') {
		if (transa == 'N' && transb == 'N') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register double sum = 0.0;
					for (int k=0; k<K; k++) {
						double tmp = A[k + m * lda] * B[n + k * ldb];
						sum += tmp;
					}
					C[n + m * ldc] = alpha * sum + beta * C[n + m * ldc];
				}
			}
		}

		if (transa == 'T' && transb == 'N') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register double sum = 0.0;
					for (int k=0; k<K; k++) {
						double tmp = A[m + k * lda] * B[n + k * ldb];
						sum += tmp;
					}
					C[n + m * ldc] = alpha * sum + beta * C[n + m * ldc];
				}
			}
		}

		if (transa == 'N' && transb == 'T') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register double sum = 0.0;
					for (int k=0; k<K; k++) {
						double tmp = A[k + m * lda] * B[k + n * ldb];
						sum += tmp;
					}
					C[n + m * ldc] = alpha * sum + beta * C[n + m * ldc];
				}
			}
		}

		if (transa == 'T' && transb == 'T') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register double sum = 0.0;
					for (int k=0; k<K; k++) {
						double tmp = A[m + k * lda] * B[k + n * ldb];
						sum += tmp;
					}
					C[n + m * ldc] = alpha * sum + beta * C[n + m * ldc];
				}
			}
		}
	} else
	// ColMajor
	/*if (major == 'C')*/ {
		if (transa == 'N' && transb == 'N') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register double sum = 0.0;
					for (int k=0; k<K; k++) {
						double tmp = A[m + k * lda] * B[k + n * ldb];
						sum += tmp;
					}
					C[m + n * ldc] = alpha * sum + beta * C[m + n * ldc];
				}
			}
		}

		if (transa == 'T' && transb == 'N') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register double sum = 0.0;
					for (int k=0; k<K; k++) {
						double tmp = A[k + m * lda] * B[k + n * ldb];
						sum += tmp;
					}
					C[m + n * ldc] = alpha * sum + beta * C[m + n * ldc];
				}
			}
		}

		if (transa == 'N' && transb == 'T') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register double sum = 0.0;
					for (int k=0; k<K; k++) {
						double tmp = A[m + k * lda] * B[n + k * ldb];
						sum += tmp;
					}
					C[m + n * ldc] = alpha * sum + beta * C[m + n * ldc];
				}
			}
		}

		if (transa == 'T' && transb == 'T') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register double sum = 0.0;
					for (int k=0; k<K; k++) {
						double tmp = A[k + m * lda] * B[n + k * ldb];
						sum += tmp;
					}
					C[m + n * ldc] = alpha * sum + beta * C[m + n * ldc];
				}
			}
		}
	}
}
void sgemm_cpu(
	char		major,
	char		transa,
	char		transb,
	const int	M,
	const int	N,
	const int	K,
	const float	alpha,
	const float	*A,
	const int	lda,
	const float	*B,
	const int	ldb,
	const float	beta,
	float		*C,
	const int	ldc)
{
	// RowMajor
	if (major == 'R') {
		if (transa == 'N' && transb == 'N') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register float sum = 0.0;
					for (int k=0; k<K; k++) {
						float tmp = A[k + m * lda] * B[n + k * ldb];
						sum += tmp;
					}
					C[n + m * ldc] = alpha * sum + beta * C[n + m * ldc];
				}
			}
		}

		if (transa == 'T' && transb == 'N') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register float sum = 0.0;
					for (int k=0; k<K; k++) {
						float tmp = A[m + k * lda] * B[n + k * ldb];
						sum += tmp;
					}
					C[n + m * ldc] = alpha * sum + beta * C[n + m * ldc];
				}
			}
		}

		if (transa == 'N' && transb == 'T') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register float sum = 0.0;
					for (int k=0; k<K; k++) {
						float tmp = A[k + m * lda] * B[k + n * ldb];
						sum += tmp;
					}
					C[n + m * ldc] = alpha * sum + beta * C[n + m * ldc];
				}
			}
		}

		if (transa == 'T' && transb == 'T') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register float sum = 0.0;
					for (int k=0; k<K; k++) {
						float tmp = A[m + k * lda] * B[k + n * ldb];
						sum += tmp;
					}
					C[n + m * ldc] = alpha * sum + beta * C[n + m * ldc];
				}
			}
		}
	} else
	// ColMajor
	/*if (major == 'C')*/ {
		if (transa == 'N' && transb == 'N') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register float sum = 0.0;
					for (int k=0; k<K; k++) {
						float tmp = A[m + k * lda] * B[k + n * ldb];
						sum += tmp;
					}
					C[m + n * ldc] = alpha * sum + beta * C[m + n * ldc];
				}
			}
		}

		if (transa == 'T' && transb == 'N') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register float sum = 0.0;
					for (int k=0; k<K; k++) {
						float tmp = A[k + m * lda] * B[k + n * ldb];
						sum += tmp;
					}
					C[m + n * ldc] = alpha * sum + beta * C[m + n * ldc];
				}
			}
		}

		if (transa == 'N' && transb == 'T') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register float sum = 0.0;
					for (int k=0; k<K; k++) {
						float tmp = A[m + k * lda] * B[n + k * ldb];
						sum += tmp;
					}
					C[m + n * ldc] = alpha * sum + beta * C[m + n * ldc];
				}
			}
		}

		if (transa == 'T' && transb == 'T') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register float sum = 0.0;
					for (int k=0; k<K; k++) {
						float tmp = A[k + m * lda] * B[n + k * ldb];
						sum += tmp;
					}
					C[m + n * ldc] = alpha * sum + beta * C[m + n * ldc];
				}
			}
		}
	}
}
/*void sgemm_cpu(
	char	major,
	char		transa,
	char		transb,
	int M, int N, int K, 
	float alpha, 
	const float *A, int lda, 
	const float *B, int ldb,
	float beta, 
	float *C, int ldc)
{
  if (beta != 0) {
    for (int m = 0; m < M; A += lda, C += ldc, ++m) {
      for (int n = 0; n < N; ++n) {
        const float *Bcol = &B[n];
        double acc = 0;
        for (int k = 0; k < K; Bcol += ldb, ++k)
          acc += (double)A[k] * Bcol[0];
        C[n] = C[n]*beta + acc*alpha;
      }
    }
  } else {
    for (int m = 0; m < M; A += lda, C += ldc, ++m) {
      for (int n = 0; n < N; ++n) {
        const float *Bcol = &B[n];
        double acc = 0;
        for (int k = 0; k < K; Bcol += ldb, ++k)
          acc += (double)A[k] * Bcol[0];
        C[n] = acc*alpha;
      }
    }
  }
}*/
