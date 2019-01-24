real dot8(const real *x, const real *y, int n)
{
	int i, n8 = n>>3<<3;
	real s, t[8];
	t[0] = t[1] = t[2] = t[3] = t[4] = t[5] = t[6] = t[7] = 0.0;
	for (i=0; i<n8; i+=8) {
		t[0] += x[i+0] * y[i+0];
		t[1] += x[i+1] * y[i+1];
		t[2] += x[i+2] * y[i+2];
		t[3] += x[i+3] * y[i+3];
		t[4] += x[i+4] * y[i+4];
		t[5] += x[i+5] * y[i+5];
		t[6] += x[i+6] * y[i+6];
		t[7] += x[i+7] * y[i+7];
	}
	for (s=0.0; i<n; i++) s += x[i] * y[i];
	return s + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
}
/*float dot_avx_2(const float *vec1, const float *vec2, int n)
{
	int i, n16 = n>>4<<4;
	__m256 u1 = {0};
	__m256 u2 = {0};
	for (i=0; i<n16; i+=16) {
		__m256 w1 = _mm256_load_ps(&vec1[i]);
		__m256 w2 = _mm256_load_ps(&vec1[i+8]);
		__m256 x1 = _mm256_load_ps(&vec2[i]);
		__m256 x2 = _mm256_load_ps(&vec2[i+8]);

		x1 = _mm256_mul_ps(w1, x1);
		x2 = _mm256_mul_ps(w2, x2);
		u1 = _mm256_add_ps(u1, x1);
		u2 = _mm256_add_ps(u2, x2);
	}
	u1 = _mm256_add_ps(u1, u2);

	__attribute__((aligned(32))) static float t[8];// = {0};
	_mm256_store_ps(t, u1);

	for (; i<n; i++) t[0] += vec1[i] * vec2[i];
	return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
}*/
float sdot8_avx256(const float *a, const float *b, int n)
{
	int i, n8 = n>>3<<3;
	register __m256 va, vb, vtemp;
	register __m128 vhigh, vresult;
	register float s = 0.0;

	for (i=0; i<n8; i+=8) {
		// load
		va = _mm256_loadu_ps(a+i); // matrix_a[i][k]
		vb = _mm256_loadu_ps(b+i); // matrix_b[j][k]

		// multiply
		vtemp = _mm256_mul_ps(va, vb);

		// add
		// extract higher four floats
		vhigh = _mm256_extractf128_ps(vtemp, 1); // high 128
		// add higher four floats to lower floats
		vresult = _mm_add_ps(_mm256_castps256_ps128(vtemp), vhigh);
		// horizontal add of that result
		vresult = _mm_hadd_ps(vresult, vresult);
		// another horizontal add of that result
		vresult = _mm_hadd_ps(vresult, vresult);

		// store
		s += _mm_cvtss_f32(vresult);
	}
	for (; i<n; i++) s += a[i] * b[i];
	return s;
}
//#define BLOCK_SIZE 50
void gemm_(
	char		major,
	char		transa,
	char		transb,
	const int	M,
	const int	N,
	const int	K,
	const real	alpha,
	const real	*A,
	const int	lda,
	const real	*B,
	const int	ldb,
	const real	beta,
	real		*C,
	const int	ldc)
{
	if (transa == 'N' && transb == 'T') {
		// RNT
		for (int m=0; m<M; m++) {
			for (int n=0; n<N; n++) {
#ifdef CATS_USE_FLOAT
//				C[m*ldc+n] = sdot8_avx256(&A[m*lda], &B[n*ldb], K);
				C[m*ldc+n] = alpha * sdot8_avx256(&A[m*lda], &B[n*ldb], K) + beta * C[m*ldc+n];
#else
				C[m*ldc+n] = alpha * dot8(&A[m*lda], &B[n*ldb], K) + beta * C[m*ldc+n];
#endif
			}
		}
	} else if (transa == 'N' && transb == 'N') {
		// RNN
		// https://www.eidos.ic.i.u-tokyo.ac.jp/~tau/lecture/parallel_distributed/2016/slides/pdf/simd.pdf
		/*for (int m=0; m<M; m+=5) {
			for (int n=0; n<N; n+=16) {
				for (int k=0; k<K; k++) {
					for (int di=0; di<5; di++) {
						for (int dj=0; dj<16; dj+=8) {
//							C[m+di,n+dj:n+dj+8] += A[m+di,k] * B[k,n+dj:j+8];
							real *a = A + (m+di)*lda + k;
							real *b = B + k*ldb + n+dj;
							real *c = C + (m+di)*ldc + n+dj;
							*c++ += *a++ * *b++;
							*c++ += *a++ * *b++;
							*c++ += *a++ * *b++;
							*c++ += *a++ * *b++;
							*c++ += *a++ * *b++;
							*c++ += *a++ * *b++;
							*c++ += *a++ * *b++;
							*c++ += *a++ * *b++;
						}
					}
				}
			}
		}*/
		/*for (int i=0; i<M; i++) {
			for(int j=0; j<N; j+=8) {
				register __m256 va, vb, vr;
				vr = _mm256_setzero_ps();
				for (int k=0; k<K; k++) {
					//result[i][j] += mat1[i][k] * mat2[k][j];
					va = _mm256_loadu_ps(A+i*lda+k);
					vb = _mm256_loadu_ps(B+k*ldb+j);
					vr = _mm_add_epi32(vR, _mm_mullo_epi32(vA, vB));
				}
				_mm_storeu_si128((__m128i*)&result[i][j], vR));
			}
		}*/
		/*for (int i=0; i<M; i++) {
			for(int j=0; j<N; j+=4) {
				// vectorize over this loop
				__m128i vR = _mm_setzero_si128();
				for (int k=0; k<K; k++) {
					//result[i][j] += mat1[i][k] * mat2[k][j];
					__m128i vA = _mm_set1_epi32(mat1[i][k]);  // load+broadcast is much cheaper than MOVD + 3 inserts (or especially 4x insert, which your new code is doing)
					__m128i vB = _mm_loadu_si128((__m128i*)&mat2[k][j]);  // mat2[k][j+0..3]
					vR = _mm_add_epi32(vR, _mm_mullo_epi32(vA, vB));
				}
				_mm_storeu_si128((__m128i*)&result[i][j], vR));
			}
		}*/
		/*for (int n=0; n<N; n++) {
			for (int i=0; i<K; i++) {
				C[n+m*ldc] = alpha * sum + beta * C[n+m*ldc];
			}
			for (int m=0; m<M; m++) {
					register real sum = 0.0;
					for (int k=0; k<K; k++) {
						sum += A[k + m * lda] * B[n + k * ldb];
					}
					C[n+m*ldc] = alpha * sum + beta * C[n+m*ldc];
			}
		}*/
	}
#if 0
//	int i, j, k;
	memset(C, 0, ldc*M*sizeof(real));
	if (transa == 'N' && transb == 'T') {
		/*for (i=0; i<M; i++) {
			for (k=0; k<K; k++) {
				for (j=0; j<N; j++) {
					C[i*ldc+j] += A[i*lda+k] * B[k*ldb+j];
				}
			}
		}*/
		for (int m=0; m<M; m++) {
			for (int n=0; n<N; n++) {
				for (int k=0; k<K; k++) {
					C[n+m*ldc] += A[k+m*lda] * B[k+n*ldb];
				}
			}
		}
	}
#endif
	/*int i, j, k, ii, jj, kk;
	for (i=0; i<N; i+=BLOCK_SIZE) {
		for (j=0; j<N; j+=BLOCK_SIZE) {
			for (k=0; k<N; k+=BLOCK_SIZE) {
				for (ii=i; ii<(i+BLOCK_SIZE); ii+=2) {
					for (jj=j; jj<(j+BLOCK_SIZE); jj++) {
						register real s0 = 0.0;
						register real s1 = 0.0;
						for (kk=k; kk<(k+BLOCK_SIZE); kk+=5) {
							s0 += a[ii][kk] * b[kk][jj];
							s0 += a[ii][kk+1] * b[kk+1][jj];
							s0 += a[ii][kk+2] * b[kk+2][jj];
							s0 += a[ii][kk+3] * b[kk+3][jj];
							s0 += a[ii][kk+4] * b[kk+4][jj];

							s1 += a[ii+1][kk] * b[kk][jj];
							s1 += a[ii+1][kk+1] * b[kk+1][jj];
							s1 += a[ii+1][kk+2] * b[kk+2][jj];
							s1 += a[ii+1][kk+3] * b[kk+3][jj];
							s1 += a[ii+1][kk+4] * b[kk+4][jj];
						}
						c[ii][jj] += s0;
						c[ii+1][jj] += s1;
					}
				}
			}
		}
	}*/
}
