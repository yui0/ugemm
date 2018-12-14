#define MC  384
#define KC  384
#define NC  4096

#define MR  8
#define NR  8

//  Local buffers for storing panels from A, B and C
static float _a[MC*KC] __attribute__ ((aligned (16)));
static float _b[KC*NC] __attribute__ ((aligned (16)));
static float _c[MR*NR] __attribute__ ((aligned (16)));

//  Packing complete panels from A (i.e. without padding)
static void _pack_MRxk(int k, const float *A, int incRowA, int incColA, float *buffer)
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

//  Packing panels from A with padding if required
static void _pack_A(int mc, int kc, const float *A, int incRowA, int incColA, float *buffer)
{
	int mp  = mc / MR;
	int _mr = mc % MR;

	int i, j;

	for (i=0; i<mp; ++i) {
		_pack_MRxk(kc, A, incRowA, incColA, buffer);
		buffer += kc*MR;
		A      += MR*incRowA;
	}
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

//  Packing complete panels from B (i.e. without padding)
static void _pack_kxNR(int k, const float *B, int incRowB, int incColB, float *buffer)
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

//  Packing panels from B with padding if required
static void _pack_B(int kc, int nc, const float *B, int incRowB, int incColB, float *buffer)
{
	int np  = nc / NR;
	int _nr = nc % NR;

	int i, j;

	for (j=0; j<np; ++j) {
		_pack_kxNR(kc, B, incRowB, incColB, buffer);
		buffer += kc*NR;
		B      += NR*incColB;
	}
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

#if 0
void matmul4x4_sse(const float *A, const float *B, float *C)
{
	__m128 row1 = _mm_load_ps(&B[0]);
	__m128 row2 = _mm_load_ps(&B[4]);
	__m128 row3 = _mm_load_ps(&B[8]);
	__m128 row4 = _mm_load_ps(&B[12]);
	for (int i=0; i<4; i++) {
		__m128 brod1 = _mm_set1_ps(A[4*i + 0]);
		__m128 brod2 = _mm_set1_ps(A[4*i + 1]);
		__m128 brod3 = _mm_set1_ps(A[4*i + 2]);
		__m128 brod4 = _mm_set1_ps(A[4*i + 3]);
		__m128 row = _mm_add_ps(
			_mm_add_ps(
				_mm_mul_ps(brod1, row1),
				_mm_mul_ps(brod2, row2)),
			_mm_add_ps(
				_mm_mul_ps(brod3, row3),
				_mm_mul_ps(brod4, row4)));
		_mm_store_ps(&C[4*i], row);
	}
}
void matmul4x4_avx(const float *a, const float *b, float *c)
{
	// Perform a 4x4 matrix multiply by a 4x4 matrix 
	register __m256 a0, a1, b0, b1;
	register __m256 c0, c1, c2, c3, c4, c5, c6, c7;
	register __m256 t0, t1, u0, u1;

//#define SIMD_ALIGNED
#ifdef SIMD_ALIGNED
	t0 = _mm256_load_ps(a);						// t0 = a00, a01, a02, a03, a10, a11, a12, a13
	t1 = _mm256_load_ps(a+8);					// t1 = a20, a21, a22, a23, a30, a31, a32, a33
	u0 = _mm256_load_ps(b);						// u0 = b00, b01, b02, b03, b10, b11, b12, b13
	u1 = _mm256_load_ps(b+8);					// u1 = b20, b21, b22, b23, b30, b31, b32, b33
#else
	t0 = _mm256_loadu_ps(a);					// t0 = a00, a01, a02, a03, a10, a11, a12, a13
	t1 = _mm256_loadu_ps(a+8);					// t1 = a20, a21, a22, a23, a30, a31, a32, a33
	u0 = _mm256_loadu_ps(b);					// u0 = b00, b01, b02, b03, b10, b11, b12, b13
	u1 = _mm256_loadu_ps(b+8);					// u1 = b20, b21, b22, b23, b30, b31, b32, b33
#endif

	a0 = _mm256_shuffle_ps(t0, t0, _MM_SHUFFLE(0, 0, 0, 0));	// a0 = a00, a00, a00, a00, a10, a10, a10, a10
	a1 = _mm256_shuffle_ps(t1, t1, _MM_SHUFFLE(0, 0, 0, 0));	// a1 = a20, a20, a20, a20, a30, a30, a30, a30
	b0 = _mm256_permute2f128_ps(u0, u0, 0x00);			// b0 = b00, b01, b02, b03, b00, b01, b02, b03  
	c0 = _mm256_mul_ps(a0, b0);					// c0 = a00*b00  a00*b01  a00*b02  a00*b03  a10*b00  a10*b01  a10*b02  a10*b03
	c1 = _mm256_mul_ps(a1, b0);					// c1 = a20*b00  a20*b01  a20*b02  a20*b03  a30*b00  a30*b01  a30*b02  a30*b03

	a0 = _mm256_shuffle_ps(t0, t0, _MM_SHUFFLE(1, 1, 1, 1));	// a0 = a01, a01, a01, a01, a11, a11, a11, a11
	a1 = _mm256_shuffle_ps(t1, t1, _MM_SHUFFLE(1, 1, 1, 1));	// a1 = a21, a21, a21, a21, a31, a31, a31, a31
	b0 = _mm256_permute2f128_ps(u0, u0, 0x11);			// b0 = b10, b11, b12, b13, b10, b11, b12, b13
	c2 = _mm256_mul_ps(a0, b0);					// c2 = a01*b10  a01*b11  a01*b12  a01*b13  a11*b10  a11*b11  a11*b12  a11*b13
	c3 = _mm256_mul_ps(a1, b0);					// c3 = a21*b10  a21*b11  a21*b12  a21*b13  a31*b10  a31*b11  a31*b12  a31*b13

	a0 = _mm256_shuffle_ps(t0, t0, _MM_SHUFFLE(2, 2, 2, 2));	// a0 = a02, a02, a02, a02, a12, a12, a12, a12
	a1 = _mm256_shuffle_ps(t1, t1, _MM_SHUFFLE(2, 2, 2, 2));	// a1 = a22, a22, a22, a22, a32, a32, a32, a32
	b1 = _mm256_permute2f128_ps(u1, u1, 0x00);			// b0 = b20, b21, b22, b23, b20, b21, b22, b23
	c4 = _mm256_mul_ps(a0, b1);					// c4 = a02*b20  a02*b21  a02*b22  a02*b23  a12*b20  a12*b21  a12*b22  a12*b23
	c5 = _mm256_mul_ps(a1, b1);					// c5 = a22*b20  a22*b21  a22*b22  a22*b23  a32*b20  a32*b21  a32*b22  a32*b23

	a0 = _mm256_shuffle_ps(t0, t0, _MM_SHUFFLE(3, 3, 3, 3));	// a0 = a03, a03, a03, a03, a13, a13, a13, a13
	a1 = _mm256_shuffle_ps(t1, t1, _MM_SHUFFLE(3, 3, 3, 3));	// a1 = a23, a23, a23, a23, a33, a33, a33, a33
	b1 = _mm256_permute2f128_ps(u1, u1, 0x11);			// b0 = b30, b31, b32, b33, b30, b31, b32, b33
	c6 = _mm256_mul_ps(a0, b1);					// c6 = a03*b30  a03*b31  a03*b32  a03*b33  a13*b30  a13*b31  a13*b32  a13*b33
	c7 = _mm256_mul_ps(a1, b1);					// c7 = a23*b30  a23*b31  a23*b32  a23*b33  a33*b30  a33*b31  a33*b32  a33*b33

	c0 = _mm256_add_ps(c0, c2);					// c0 = c0 + c2 (two terms, first two rows)
	c4 = _mm256_add_ps(c4, c6);					// c4 = c4 + c6 (the other two terms, first two rows)
	c1 = _mm256_add_ps(c1, c3);					// c1 = c1 + c3 (two terms, second two rows)
	c5 = _mm256_add_ps(c5, c7);					// c5 = c5 + c7 (the other two terms, second two rose)

	// Finally complete addition of all four terms and return the results
#ifdef SIMD_ALIGNED
	_mm256_store_ps(c, _mm256_add_ps(c0, c4));			// n0 = a00*b00+a01*b10+a02*b20+a03*b30  a00*b01+a01*b11+a02*b21+a03*b31  a00*b02+a01*b12+a02*b22+a03*b32  a00*b03+a01*b13+a02*b23+a03*b33
									//      a10*b00+a11*b10+a12*b20+a13*b30  a10*b01+a11*b11+a12*b21+a13*b31  a10*b02+a11*b12+a12*b22+a13*b32  a10*b03+a11*b13+a12*b23+a13*b33
	_mm256_store_ps(c+8, _mm256_add_ps(c1, c5));			// n1 = a20*b00+a21*b10+a22*b20+a23*b30  a20*b01+a21*b11+a22*b21+a23*b31  a20*b02+a21*b12+a22*b22+a23*b32  a20*b03+a21*b13+a22*b23+a23*b33
									//      a30*b00+a31*b10+a32*b20+a33*b30  a30*b01+a31*b11+a32*b21+a33*b31  a30*b02+a31*b12+a32*b22+a33*b32  a30*b03+a31*b13+a32*b23+a33*b33
#else
	_mm256_storeu_ps(c, _mm256_add_ps(c0, c4));			// n0 = a00*b00+a01*b10+a02*b20+a03*b30  a00*b01+a01*b11+a02*b21+a03*b31  a00*b02+a01*b12+a02*b22+a03*b32  a00*b03+a01*b13+a02*b23+a03*b33
									//      a10*b00+a11*b10+a12*b20+a13*b30  a10*b01+a11*b11+a12*b21+a13*b31  a10*b02+a11*b12+a12*b22+a13*b32  a10*b03+a11*b13+a12*b23+a13*b33
	_mm256_storeu_ps(c+8, _mm256_add_ps(c1, c5));			// n1 = a20*b00+a21*b10+a22*b20+a23*b30  a20*b01+a21*b11+a22*b21+a23*b31  a20*b02+a21*b12+a22*b22+a23*b32  a20*b03+a21*b13+a22*b23+a23*b33
									//      a30*b00+a31*b10+a32*b20+a33*b30  a30*b01+a31*b11+a32*b21+a33*b31  a30*b02+a31*b12+a32*b22+a33*b32  a30*b03+a31*b13+a32*b23+a33*b33
#endif
	return;
}
#endif
/*void dot8x8_avx(const float *a, const float *b, float *c)
{
	register __m256 a0, b0, b1, b2, b3, b4, b5, b6, b7;
	register __m256 c0, c1, c2, c3, c4, c5, c6, c7;
	b0 = _mm256_broadcast_ss(b);
	b1 = _mm256_broadcast_ss(b+1);
	b2 = _mm256_broadcast_ss(b+2);
	b3 = _mm256_broadcast_ss(b+3);
	b4 = _mm256_broadcast_ss(b+4);
	b5 = _mm256_broadcast_ss(b+5);
	b6 = _mm256_broadcast_ss(b+6);
	b7 = _mm256_broadcast_ss(b+7);
	a0 = _mm256_loadu_ps(a);
	c0 = _mm256_loadu_ps(c);
	c1 = _mm256_loadu_ps(c+8);
	c2 = _mm256_loadu_ps(c+16);
	c3 = _mm256_loadu_ps(c+24);
	c4 = _mm256_loadu_ps(c+32);
	c5 = _mm256_loadu_ps(c+40);
	c6 = _mm256_loadu_ps(c+48);
	c7 = _mm256_loadu_ps(c+56);
	_mm256_storeu_ps(c, _mm256_add_ps(c0, _mm256_mul_ps(a0, b0)));
	_mm256_storeu_ps(c+8, _mm256_add_ps(c1, _mm256_mul_ps(a0, b1)));
	_mm256_storeu_ps(c+16, _mm256_add_ps(c2, _mm256_mul_ps(a0, b2)));
	_mm256_storeu_ps(c+24, _mm256_add_ps(c3, _mm256_mul_ps(a0, b3)));
	_mm256_storeu_ps(c+32, _mm256_add_ps(c4, _mm256_mul_ps(a0, b4)));
	_mm256_storeu_ps(c+40, _mm256_add_ps(c5, _mm256_mul_ps(a0, b5)));
	_mm256_storeu_ps(c+48, _mm256_add_ps(c6, _mm256_mul_ps(a0, b6)));
	_mm256_storeu_ps(c+56, _mm256_add_ps(c7, _mm256_mul_ps(a0, b7)));
}*/
void dot8x8_avx(const float *a, const float *b, float *c, int kc)
{
	register __m256 a0, b0, b1, b2, b3, b4, b5, b6, b7;
	register __m256 c0, c1, c2, c3, c4, c5, c6, c7;
	c0 = _mm256_loadu_ps(c);
	c1 = _mm256_loadu_ps(c+8);
	c2 = _mm256_loadu_ps(c+16);
	c3 = _mm256_loadu_ps(c+24);
	c4 = _mm256_loadu_ps(c+32);
	c5 = _mm256_loadu_ps(c+40);
	c6 = _mm256_loadu_ps(c+48);
	c7 = _mm256_loadu_ps(c+56);

	for (int i=0; i<kc; i++) {
		b0 = _mm256_broadcast_ss(b);
		b1 = _mm256_broadcast_ss(b+1);
		b2 = _mm256_broadcast_ss(b+2);
		b3 = _mm256_broadcast_ss(b+3);
		b4 = _mm256_broadcast_ss(b+4);
		b5 = _mm256_broadcast_ss(b+5);
		b6 = _mm256_broadcast_ss(b+6);
		b7 = _mm256_broadcast_ss(b+7);
		a0 = _mm256_loadu_ps(a);

		c0 = _mm256_add_ps(c0, _mm256_mul_ps(a0, b0));
		c1 = _mm256_add_ps(c1, _mm256_mul_ps(a0, b1));
		c2 = _mm256_add_ps(c2, _mm256_mul_ps(a0, b2));
		c3 = _mm256_add_ps(c3, _mm256_mul_ps(a0, b3));
		c4 = _mm256_add_ps(c4, _mm256_mul_ps(a0, b4));
		c5 = _mm256_add_ps(c5, _mm256_mul_ps(a0, b5));
		c6 = _mm256_add_ps(c6, _mm256_mul_ps(a0, b6));
		c7 = _mm256_add_ps(c7, _mm256_mul_ps(a0, b7));

		a += 8;
		b += 8;
	}

	_mm256_storeu_ps(c, c0);
	_mm256_storeu_ps(c+8, c1);
	_mm256_storeu_ps(c+16, c2);
	_mm256_storeu_ps(c+24, c3);
	_mm256_storeu_ps(c+32, c4);
	_mm256_storeu_ps(c+40, c5);
	_mm256_storeu_ps(c+48, c6);
	_mm256_storeu_ps(c+56, c7);
}
//  Micro kernel for multiplying panels from A and B.
static void _sgemm_micro_kernel(
	long kc,
	float alpha, const float *A, const float *B,
	float beta,
	float *C, long incRowC, long incColC)
{
	static float AB[MR*NR] __attribute__ ((aligned (16)));
	int i, j;

	//  Compute AB = A*B
	memset(AB, 0, MR*NR*sizeof(float));
	dot8x8_avx(A, B, AB, kc);
#if 0
	for (int l=0; l<kc; ++l) {
//		matmul4x4_sse(A, B, AB);
//		matmul4x4_avx(A, B, AB);
		dot8x8_avx(A, B, AB);
		/*for (j=0; j<NR; ++j) {
			for (i=0; i<MR; ++i) {
				AB[i+j*MR] += A[i]*B[j];
			}
		}*/
		A += MR;
		B += NR;
	}
#endif

	//  Update C <- beta*C
	if (beta==0.0) {
		for (j=0; j<NR; ++j) {
			for (i=0; i<MR; ++i) {
				C[i*incRowC+j*incColC] = 0.0;
			}
		}
	} else if (beta!=1.0) {
		for (j=0; j<NR; ++j) {
			for (i=0; i<MR; ++i) {
				C[i*incRowC+j*incColC] *= beta;
			}
		}
	}

	//  Update C <- C + alpha*AB (note: the case alpha==0.0 was already treated in
	//                                  the above layer sgemm_nn)
	if (alpha==1.0) {
		for (j=0; j<NR; ++j) {
			for (i=0; i<MR; ++i) {
				C[i*incRowC+j*incColC] += AB[i+j*MR];
			}
		}
	} else {
		for (j=0; j<NR; ++j) {
			for (i=0; i<MR; ++i) {
				C[i*incRowC+j*incColC] += alpha*AB[i+j*MR];
			}
		}
	}
}

//  Compute Y += alpha*X
static void sgeaxpy(
	int           m,
	int           n,
	float        alpha,
	const float  *X,
	int           incRowX,
	int           incColX,
	float        *Y,
	int           incRowY,
	int           incColY)
{
	int i, j;

	if (alpha!=1.0) {
		for (j=0; j<n; ++j) {
			for (i=0; i<m; ++i) {
				Y[i*incRowY+j*incColY] += alpha*X[i*incRowX+j*incColX];
			}
		}
	} else {
		for (j=0; j<n; ++j) {
			for (i=0; i<m; ++i) {
				Y[i*incRowY+j*incColY] += X[i*incRowX+j*incColX];
			}
		}
	}
}

//  Compute X *= alpha
static void sgescal(int m, int n, float alpha, float *X, int incRowX, int incColX)
{
	int i, j;

	if (alpha!=0.0) {
		for (j=0; j<n; ++j) {
			for (i=0; i<m; ++i) {
				X[i*incRowX+j*incColX] *= alpha;
			}
		}
	} else {
		for (j=0; j<n; ++j) {
			for (i=0; i<m; ++i) {
				X[i*incRowX+j*incColX] = 0.0;
			}
		}
	}
}

//  Macro Kernel for the multiplication of blocks of A and B.  We assume that
//  these blocks were previously packed to buffers _A and _B.
static void _sgemm_macro_kernel(
	int     mc,
	int     nc,
	int     kc,
	float  alpha,
	float  beta,
	float  *C,
	int     incRowC,
	int     incColC)
{
	int mp = (mc+MR-1) / MR;
	int np = (nc+NR-1) / NR;

	int _mr = mc % MR;
	int _nr = nc % NR;

	int mr, nr;
	int i, j;

	for (j=0; j<np; ++j) {
		nr    = (j!=np-1 || _nr==0) ? NR : _nr;

		for (i=0; i<mp; ++i) {
			mr    = (i!=mp-1 || _mr==0) ? MR : _mr;

			if (mr==MR && nr==NR) {
				_sgemm_micro_kernel(kc, alpha, &_a[i*kc*MR], &_b[j*kc*NR],
				                   beta, &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
			} else {
				_sgemm_micro_kernel(kc, alpha, &_a[i*kc*MR], &_b[j*kc*NR], 0.0, _c, 1, MR);
				sgescal(mr, nr, beta, &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
				sgeaxpy(mr, nr, 1.0, _c, 1, MR, &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
			}
		}
	}
}

//  Compute C <- beta*C + alpha*A*B
void sgemm_nn(
	int m, int n, int k, float alpha,
	const float *A, int incRowA, int incColA, const float *B, int incRowB, int incColB,
	float beta, float *C, int incRowC, int incColC)
{
	int mb = (m+MC-1) / MC;
	int nb = (n+NC-1) / NC;
	int kb = (k+KC-1) / KC;

	int _mc = m % MC;
	int _nc = n % NC;
	int _kc = k % KC;

	int mc, nc, kc;
	int i, j, l;

	float _beta;

	if (alpha==0.0 || k==0) {
		sgescal(m, n, beta, C, incRowC, incColC);
		return;
	}

	for (j=0; j<nb; ++j) {
		nc = (j!=nb-1 || _nc==0) ? NC : _nc;

		for (l=0; l<kb; ++l) {
			kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;
			_beta = (l==0) ? beta : 1.0;

			_pack_B(kc, nc, &B[l*KC*incRowB+j*NC*incColB], incRowB, incColB, _b);

			for (i=0; i<mb; ++i) {
				mc = (i!=mb-1 || _mc==0) ? MC : _mc;

				_pack_A(mc, kc, &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA, _a);

				_sgemm_macro_kernel(mc, nc, kc, alpha, _beta, &C[i*MC*incRowC+j*NC*incColC], incRowC, incColC);
			}
		}
	}
}

void sgemm_sse(
	char		major,
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
	if (major == 'C') {
		if (transB=='N') {
			if (transA=='N') {
				// Form  C := alpha*A*B + beta*C
				sgemm_nn(m, n, k, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
			} else {
				// Form  C := alpha*A**T*B + beta*C
				sgemm_nn(m, n, k, alpha, A, ldA, 1, B, 1, ldB, beta, C, 1, ldC);
			}
		} else {
			if (transA=='N') {
				// Form  C := alpha*A*B**T + beta*C
				sgemm_nn(m, n, k, alpha, A, 1, ldA, B, ldB, 1, beta, C, 1, ldC);
			} else {
				// Form  C := alpha*A**T*B**T + beta*C
				sgemm_nn(m, n, k, alpha, A, ldA, 1, B, ldB, 1, beta, C, 1, ldC);
			}
		}
	} else {
		if (transB=='N') {
			if (transA=='N') {
				// Form  C := alpha*A*B + beta*C
				sgemm_nn(m, n, k, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
			} else {
				// Form  C := alpha*A**T*B + beta*C
				sgemm_nn(m, n, k, alpha, A, 1, ldA, B, ldB, 1, beta, C, ldC, 1);
			}
		} else {
			if (transA=='N') {
				// Form  C := alpha*A*B**T + beta*C
				sgemm_nn(m, n, k, alpha, A, ldA, 1, B, 1, ldB, beta, C, ldC, 1);
			} else {
				// Form  C := alpha*A**T*B**T + beta*C
				sgemm_nn(m, n, k, alpha, A, 1, ldA, B, 1, ldB, beta, C, ldC, 1);
			}
		}
	}
}

#undef MC
#undef KC
#undef NC

#undef MR
#undef NR
