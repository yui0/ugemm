// clang -Ofast -o axpy axpy.c -march=native

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#if defined(__STDC_VERSION__) && __STDC_VERSION__ < 201102L
#ifdef _MSC_VER
#define _Alignas(n)  __declspec(align(n))
#else
#define _Alignas(n)  __attribute__((aligned(n)))
#endif  // _MSC_VER
#endif  // defined(__cplusplus) && __cplusplus < 201103L

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h>
#else
#include <stdlib.h>
#endif  /* defined(_MSC_VER) || defined(__MINGW32__) */
 
#ifndef __cplusplus
#if defined(_MSC_VER)
#define inline      __inline
#define __inline__  __inline
#elif !defined(__GNUC__) && !defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L
#define inline
#define __inline
#endif
#endif

static inline void* amalloc(size_t size, size_t alignment)
{
#if defined(_MSC_VER) || defined(__MINGW32__)
	return _aligned_malloc(size, alignment);
#else
	void* p;
	return posix_memalign((void**) &p, alignment, size) == 0 ? p : NULL;
#endif  /* _MSC_VER */
}

static inline void afree(void* ptr)
{
#if defined(_MSC_VER) || defined(__MINGW32__)
	_aligned_free(ptr);
#else
	free(ptr);
#endif  /* _MSC_VER */
}

void axpy(int N, double a, double* const restrict x, double* const restrict y, double* restrict z)
{
	int i;
	__m256d	const va = _mm256_broadcast_sd(&a);
	#pragma omp parallel for
	for (i=0; i<N; i+=4) { // 倍精度なので SIMD 長が 4
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


#include <stdio.h>
#include <time.h>
#define real	double
real *random_matrix(int rows, int cols)
{
	real *m = amalloc(rows*cols*sizeof(real), 32);
	for (int i=0; i<rows*cols; i++) {
		m[i] = (real)rand()/RAND_MAX;
	}
	return m;
}
void time_random_axpy(int n)
{
	real *a = random_matrix(n, 1);
	real *b = random_matrix(n, 1);
	real *c = random_matrix(n, 1);
	clock_t start = clock();
	for (int i=0; i<10; i++) {
		axpy(n, 2.0, a, b, c);
	}
	clock_t end = clock();
	printf("axpy Multiplication %d: %lf ms\n", n, (real)(end-start)/CLOCKS_PER_SEC);
	free(a);
	free(b);
	free(c);
}

int main()
{
	time_random_axpy(10000);
	time_random_axpy(100000);
	time_random_axpy(1000000);
}
