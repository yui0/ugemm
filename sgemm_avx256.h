enum {
	SIMD_FACTOR         = 8,
	COLS_PER_LOOP       = 3,
	COLS_STEPS_PER_CORE = 4,
	SIMD_ELEM_PEC_COL   = COLS_PER_LOOP*COLS_STEPS_PER_CORE,
	bb_nCols            = SIMD_ELEM_PEC_COL*SIMD_FACTOR,
	bb_nRows            = 35,
	cc_nRows            = 32,
};

typedef struct _noncblas_sgemm_prm_t {
	__m256 bb[SIMD_ELEM_PEC_COL*bb_nRows];
	__m256 cc[cc_nRows*SIMD_ELEM_PEC_COL];
	int M;
	int lda;
	int ldc;
	float alpha;
} noncblas_sgemm_prm_t;

static void avx256_noncblas_sgemm_core(
        const noncblas_sgemm_prm_t* pPrm,
        const float *A,
        float *C)
{
	int lda = pPrm->lda;
	int ldc = pPrm->ldc;
	int m;
	for (m = 0; m < pPrm->M-1; A += lda*2, C += ldc*2, m += 2) {
		float* Crow0 = C;
		float* Crow1 = C+ldc;
		for (int n = 0; n < SIMD_ELEM_PEC_COL; n += COLS_PER_LOOP) {
			const __m256 *Bcol = &pPrm->bb[n];
			__m256 a0 = _mm256_broadcast_ss(&A[0]);
			__m256 a1 = _mm256_broadcast_ss(&A[lda]);
			__m256 b;
			b = Bcol[0];
			__m256 acc00 = _mm256_mul_ps(a0, b);
			__m256 acc01 = _mm256_mul_ps(a1, b);

			b = Bcol[1];
			__m256 acc10 = _mm256_mul_ps(a0, b);
			__m256 acc11 = _mm256_mul_ps(a1, b);

			b = Bcol[2];
			__m256 acc20 = _mm256_mul_ps(a0, b);
			__m256 acc21 = _mm256_mul_ps(a1, b);

			for (int k = 1; k < bb_nRows; k += 2) {
				Bcol += SIMD_ELEM_PEC_COL;
				a0 = _mm256_broadcast_ss(&A[k]);
				a1 = _mm256_broadcast_ss(&A[k+lda]);

				b = Bcol[0];
				acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b));
				acc01 = _mm256_add_ps(acc01, _mm256_mul_ps(a1, b));

				b = Bcol[1];
				acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a0, b));
				acc11 = _mm256_add_ps(acc11, _mm256_mul_ps(a1, b));

				b = Bcol[2];
				acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a0, b));
				acc21 = _mm256_add_ps(acc21, _mm256_mul_ps(a1, b));

				Bcol += SIMD_ELEM_PEC_COL;
				a0 = _mm256_broadcast_ss(&A[k+1]);
				a1 = _mm256_broadcast_ss(&A[k+lda+1]);

				b = Bcol[0];
				acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b));
				acc01 = _mm256_add_ps(acc01, _mm256_mul_ps(a1, b));

				b = Bcol[1];
				acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a0, b));
				acc11 = _mm256_add_ps(acc11, _mm256_mul_ps(a1, b));

				b = Bcol[2];
				acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a0, b));
				acc21 = _mm256_add_ps(acc21, _mm256_mul_ps(a1, b));
			}
			__m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);

			_mm256_storeu_ps(&Crow0[SIMD_FACTOR*0], _mm256_add_ps(_mm256_mul_ps(acc00, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR*0])));
			_mm256_storeu_ps(&Crow0[SIMD_FACTOR*1], _mm256_add_ps(_mm256_mul_ps(acc10, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR*1])));
			_mm256_storeu_ps(&Crow0[SIMD_FACTOR*2], _mm256_add_ps(_mm256_mul_ps(acc20, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR*2])));

			_mm256_storeu_ps(&Crow1[SIMD_FACTOR*0], _mm256_add_ps(_mm256_mul_ps(acc01, alpha_ps), _mm256_loadu_ps(&Crow1[SIMD_FACTOR*0])));
			_mm256_storeu_ps(&Crow1[SIMD_FACTOR*1], _mm256_add_ps(_mm256_mul_ps(acc11, alpha_ps), _mm256_loadu_ps(&Crow1[SIMD_FACTOR*1])));
			_mm256_storeu_ps(&Crow1[SIMD_FACTOR*2], _mm256_add_ps(_mm256_mul_ps(acc21, alpha_ps), _mm256_loadu_ps(&Crow1[SIMD_FACTOR*2])));

			Crow0 += COLS_PER_LOOP*SIMD_FACTOR;
			Crow1 += COLS_PER_LOOP*SIMD_FACTOR;
		}
	}
	if (m < pPrm->M) {
		float* Crow0 = C;
		for (int n = 0; n < SIMD_ELEM_PEC_COL; n += COLS_PER_LOOP) {
			const __m256 *Bcol = &pPrm->bb[n];
			__m256 acc00 = _mm256_setzero_ps();
			__m256 acc10 = _mm256_setzero_ps();
			__m256 acc20 = _mm256_setzero_ps();
			for (int k = 0; k < bb_nRows; ++k) {
				__m256 a0 = _mm256_broadcast_ss(&A[k]);
				__m256 b;

				b = Bcol[0];
				acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b));

				b = Bcol[1];
				acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a0, b));

				b = Bcol[2];
				acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a0, b));
				Bcol += SIMD_ELEM_PEC_COL;
			}
			__m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);

			_mm256_storeu_ps(&Crow0[SIMD_FACTOR*0], _mm256_add_ps(_mm256_mul_ps(acc00, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR*0])));
			_mm256_storeu_ps(&Crow0[SIMD_FACTOR*1], _mm256_add_ps(_mm256_mul_ps(acc10, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR*1])));
			_mm256_storeu_ps(&Crow0[SIMD_FACTOR*2], _mm256_add_ps(_mm256_mul_ps(acc20, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR*2])));

			Crow0 += COLS_PER_LOOP*SIMD_FACTOR;
		}
	}
}

static void avx256_noncblas_sgemm_core_bottomRows(
        const noncblas_sgemm_prm_t* pPrm,
        const float *A,
        float *C,
        int nRows)
{
	int lda = pPrm->lda;
	int ldc = pPrm->ldc;
	int m;
	for (m = 0; m < pPrm->M-1; A += lda*2, C += ldc*2, m += 2) {
		float* Crow0 = C;
		float* Crow1 = C+ldc;
		for (int n = 0; n < SIMD_ELEM_PEC_COL; n += COLS_PER_LOOP) {
			const __m256 *Bcol = &pPrm->bb[n];
			__m256 acc00 = _mm256_setzero_ps();
			__m256 acc01 = _mm256_setzero_ps();
			__m256 acc10 = _mm256_setzero_ps();
			__m256 acc11 = _mm256_setzero_ps();
			__m256 acc20 = _mm256_setzero_ps();
			__m256 acc21 = _mm256_setzero_ps();
			for (int k = 0; k < nRows; ++k) {
				__m256 a0 = _mm256_broadcast_ss(&A[k]);
				__m256 a1 = _mm256_broadcast_ss(&A[k+lda]);
				__m256 b;

				b = Bcol[0];
				acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b));
				acc01 = _mm256_add_ps(acc01, _mm256_mul_ps(a1, b));

				b = Bcol[1];
				acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a0, b));
				acc11 = _mm256_add_ps(acc11, _mm256_mul_ps(a1, b));

				b = Bcol[2];
				acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a0, b));
				acc21 = _mm256_add_ps(acc21, _mm256_mul_ps(a1, b));
				Bcol += SIMD_ELEM_PEC_COL;
			}
			__m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);

			_mm256_storeu_ps(&Crow0[SIMD_FACTOR*0], _mm256_add_ps(_mm256_mul_ps(acc00, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR*0])));
			_mm256_storeu_ps(&Crow0[SIMD_FACTOR*1], _mm256_add_ps(_mm256_mul_ps(acc10, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR*1])));
			_mm256_storeu_ps(&Crow0[SIMD_FACTOR*2], _mm256_add_ps(_mm256_mul_ps(acc20, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR*2])));

			_mm256_storeu_ps(&Crow1[SIMD_FACTOR*0], _mm256_add_ps(_mm256_mul_ps(acc01, alpha_ps), _mm256_loadu_ps(&Crow1[SIMD_FACTOR*0])));
			_mm256_storeu_ps(&Crow1[SIMD_FACTOR*1], _mm256_add_ps(_mm256_mul_ps(acc11, alpha_ps), _mm256_loadu_ps(&Crow1[SIMD_FACTOR*1])));
			_mm256_storeu_ps(&Crow1[SIMD_FACTOR*2], _mm256_add_ps(_mm256_mul_ps(acc21, alpha_ps), _mm256_loadu_ps(&Crow1[SIMD_FACTOR*2])));

			Crow0 += COLS_PER_LOOP*SIMD_FACTOR;
			Crow1 += COLS_PER_LOOP*SIMD_FACTOR;
		}
	}
	if (m < pPrm->M) {
		float* Crow0 = C;
		for (int n = 0; n < SIMD_ELEM_PEC_COL; n += COLS_PER_LOOP) {
			const __m256 *Bcol = &pPrm->bb[n];
			__m256 acc00 = _mm256_setzero_ps();
			__m256 acc10 = _mm256_setzero_ps();
			__m256 acc20 = _mm256_setzero_ps();
			for (int k = 0; k < nRows; ++k) {
				__m256 a0 = _mm256_broadcast_ss(&A[k]);
				__m256 b;

				b = Bcol[0];
				acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b));

				b = Bcol[1];
				acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a0, b));

				b = Bcol[2];
				acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a0, b));
				Bcol += SIMD_ELEM_PEC_COL;
			}
			__m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);

			_mm256_storeu_ps(&Crow0[SIMD_FACTOR*0], _mm256_add_ps(_mm256_mul_ps(acc00, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR*0])));
			_mm256_storeu_ps(&Crow0[SIMD_FACTOR*1], _mm256_add_ps(_mm256_mul_ps(acc10, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR*1])));
			_mm256_storeu_ps(&Crow0[SIMD_FACTOR*2], _mm256_add_ps(_mm256_mul_ps(acc20, alpha_ps), _mm256_loadu_ps(&Crow0[SIMD_FACTOR*2])));

			Crow0 += COLS_PER_LOOP*SIMD_FACTOR;
		}
	}
}

static void avx256_noncblas_sgemm_core_rightmostColumns(
        noncblas_sgemm_prm_t* pPrm,
        const float *A,
        float *C,
        int nCols, // 0 < nCols <  bb_nCols
        int nRows) // nRows <= bb_nRows
{
	int lda = pPrm->lda;
	int ldc = pPrm->ldc;
	int ldcc = ((nCols-1)/(COLS_PER_LOOP*SIMD_FACTOR) + 1)*COLS_PER_LOOP;

	for (int m0 = 0; m0 < pPrm->M; m0 += cc_nRows) {
		int mLast = m0 + cc_nRows <= pPrm->M ? m0 + cc_nRows : pPrm->M;
		// calculate partial results and store in cc
		__m256* pCc = pPrm->cc;
		int mLastEv = mLast & (-2);
		for (int m = m0; m < mLastEv; A += lda*2, m += 2) {
			for (int n = 0; n < ldcc; n += COLS_PER_LOOP) {
				const __m256 *Bcol = &pPrm->bb[n];
				__m256 acc00 = _mm256_setzero_ps();
				__m256 acc01 = _mm256_setzero_ps();
				__m256 acc10 = _mm256_setzero_ps();
				__m256 acc11 = _mm256_setzero_ps();
				__m256 acc20 = _mm256_setzero_ps();
				__m256 acc21 = _mm256_setzero_ps();
				for (int k = 0; k < nRows; ++k) {
					__m256 a0 = _mm256_broadcast_ss(&A[k]);
					__m256 a1 = _mm256_broadcast_ss(&A[k+lda]);
					__m256 b;

					b = Bcol[0];
					acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b));
					acc01 = _mm256_add_ps(acc01, _mm256_mul_ps(a1, b));

					b = Bcol[1];
					acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a0, b));
					acc11 = _mm256_add_ps(acc11, _mm256_mul_ps(a1, b));

					b = Bcol[2];
					acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a0, b));
					acc21 = _mm256_add_ps(acc21, _mm256_mul_ps(a1, b));

					Bcol += SIMD_ELEM_PEC_COL;
				}
				__m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);

				pCc[0]      = _mm256_mul_ps(acc00, alpha_ps);
				pCc[1]      = _mm256_mul_ps(acc10, alpha_ps);
				pCc[2]      = _mm256_mul_ps(acc20, alpha_ps);
				pCc[ldcc+0] = _mm256_mul_ps(acc01, alpha_ps);
				pCc[ldcc+1] = _mm256_mul_ps(acc11, alpha_ps);
				pCc[ldcc+2] = _mm256_mul_ps(acc21, alpha_ps);
				pCc += COLS_PER_LOOP;
			}
			pCc += ldcc;
		}
		if ((mLast & 1) != 0) {
			// last row of A
			for (int n = 0; n < ldcc; n += COLS_PER_LOOP) {
				const __m256 *Bcol = &pPrm->bb[n];
				__m256 acc00 = _mm256_setzero_ps();
				__m256 acc10 = _mm256_setzero_ps();
				__m256 acc20 = _mm256_setzero_ps();
				for (int k = 0; k < nRows; ++k) {
					__m256 a0 = _mm256_broadcast_ss(&A[k]);
					__m256 b;

					b = Bcol[0];
					acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b));

					b = Bcol[1];
					acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a0, b));

					b = Bcol[2];
					acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a0, b));

					Bcol += SIMD_ELEM_PEC_COL;
				}
				__m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);

				pCc[0] = _mm256_mul_ps(acc00, alpha_ps);
				pCc[1] = _mm256_mul_ps(acc10, alpha_ps);
				pCc[2] = _mm256_mul_ps(acc20, alpha_ps);
				pCc += COLS_PER_LOOP;
			}
		}
		// add partial result in cc to C
		pCc = pPrm->cc;
		for (int m = 0; m < mLast-m0; C += ldc, pCc += ldcc, ++m) {
			const float* res = (const float*)pCc;
			for (int n = 0; n < nCols; ++n) {
				C[n] += res[n];
			}
		}
	}
}

void avx256_noncblas_sgemm(
        int M, int N, int K,
        float alpha,
        const float *A, int lda,
        const float *B, int ldb,
        float beta,
        float *C, int ldc)
{
	float *_C = C;
	if (beta != 0) {
		for (int m=0; m<M; m++) {
			for (int n=0; n<N; n++) {
				_C[n] *= beta;
			}
			_C += ldc;
		}
	} else {
		for (int m=0; m<M; m++) {
			for (int n=0; n<N; n++) {
				_C[n] = 0;
			}
			_C += ldc;
		}
	}

//	noncblas_sgemm_prm_t prm;
	noncblas_sgemm_prm_t __attribute__((aligned(32))) prm;
	prm.M = M;
	prm.lda = lda;
	prm.ldc = ldc;
	prm.alpha = alpha;

	int n_Rsteps = K / bb_nRows;
	int n_Csteps = N / bb_nCols;
	int row = 0;
	for (int ri=0; ri < n_Rsteps; ++ri) {
		int col = 0;
		for (int ci=0; ci < n_Csteps; ++ci) {
			// process full rectangles
			const float* bSrc = &B[row*ldb + col];
			#pragma unroll
			for (int i=0; i < bb_nRows; ++i) {
				memcpy(&prm.bb[SIMD_ELEM_PEC_COL*i], bSrc, bb_nCols*sizeof(*B));
				bSrc += ldb;
			}
			avx256_noncblas_sgemm_core(&prm, &A[row], &C[col]);
			col += bb_nCols;
		}
		if (col < N) {
			// process rightmost rectangle of the full-height band
			const float* bSrc = &B[row*ldb + col];
			for (int i=0; i < bb_nRows; ++i) {
				memcpy(&prm.bb[SIMD_ELEM_PEC_COL*i], bSrc, (N-col)*sizeof(*B));
				bSrc += ldb;
			}
			avx256_noncblas_sgemm_core_rightmostColumns(&prm, &A[row], &C[col], N-col, bb_nRows);
		}
		row += bb_nRows;
	}
	if (row < K) {
		// bottom band
		int col = 0;
		for (int ci=0; ci < n_Csteps; ++ci) {
			// process full-width rectangles
			const float* bSrc = &B[row*ldb + col];
			for (int i=0; i < K-row; ++i) {
				memcpy(&prm.bb[SIMD_ELEM_PEC_COL*i], bSrc, bb_nCols*sizeof(*B));
				bSrc += ldb;
			}
			avx256_noncblas_sgemm_core_bottomRows(&prm, &A[row], &C[col], K-row);
			col += bb_nCols;
		}
		if (col < N) {
			// process bottom-right corner rectangle
			const float* bSrc = &B[row*ldb + col];
			for (int i=0; i < K-row; ++i) {
				memcpy(&prm.bb[SIMD_ELEM_PEC_COL*i], bSrc, (N-col)*sizeof(*B));
				bSrc += ldb;
			}
			avx256_noncblas_sgemm_core_rightmostColumns(&prm, &A[row], &C[col], N-col, K-row);
		}
	}
}

void sgemm_avx(char	major,
	char		transA,
	char		transB,
	const int	m,
	const int	n,
	const int	k,
	const float	alpha,
	const float	*A,
	const int	ldA,
	const float	*B,
	const int	ldB,
	const float	beta,
	float		*C,
	const int	ldC)
{
	int i, j;

	//  Quick return if possible
	if (m==0 || n==0 || ((alpha==0.0 || k==0) && (beta==1.0))) {
		return;
	}

	//  And if alpha is exactly zero
	if (alpha==0.0) {
		if (beta==0.0) {
			for (j=0; j<n; j++) {
				for (i=0; i<m; i++) {
					C[i+j*ldC] = 0.0;
				}
			}
		} else {
			for (j=0; j<n; j++) {
				for (i=0; i<m; i++) {
					C[i+j*ldC] *= beta;
				}
			}
		}
		return;
	}

	//  Start the operations
	if (transB=='N') {
		if (transA=='N') {
			// Form  C := alpha*A*B + beta*C
//			dgemm_nn(m, n, k, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
			avx256_noncblas_sgemm(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
		} else {
			// Form  C := alpha*A**T*B + beta*C
//			dgemm_nn(m, n, k, alpha, A, ldA, 1, B, 1, ldB, beta, C, 1, ldC);
//			avx256_noncblas_sgemm(m, n, k, alpha, A, ldA, 1, B, 1, ldB, beta, C, 1, ldC);
//			transpose_block_SSE4x4(float *A, float *B, const int n, const int m, const int lda, const int ldb, const int block_size)
		}
	} else {
		if (transA=='N') {
			// Form  C := alpha*A*B**T + beta*C
//			dgemm_nn(m, n, k, alpha, A, 1, ldA, B, ldB, 1, beta, C, 1, ldC);
//			avx256_noncblas_sgemm(m, n, k, alpha, A, 1, ldA, B, ldB, 1, beta, C, 1, ldC);
		} else {
			// Form  C := alpha*A**T*B**T + beta*C
//			dgemm_nn(m, n, k, alpha, A, ldA, 1, B, ldB, 1, beta, C, 1, ldC);
//			avx256_noncblas_sgemm(m, n, k, alpha, A, ldA, 1, B, ldB, 1, beta, C, 1, ldC);
		}
	}
}

