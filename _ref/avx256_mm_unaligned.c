// gcc -O4 -fopt-info-optall-optimized -mavx -o avx256_mm_unaligned avx256_mm_unaligned.c
// time ./avx256_mm_unaligned > /dev/null
// https://blog.qiqitori.com/?p=398
#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    float *matrix_a = malloc(1024*1024*sizeof(float));
    float *matrix_b = malloc(1024*1024*sizeof(float));
    float result[1024][1024];
    __m256 va, vb, vtemp;
    __m128 vlow, vhigh, vresult;

    // initialize matrix_a and matrix_b
    for (int i = 0; i < 1048576; i++) {
        *(matrix_a+i) = 0.1f;
        *(matrix_b+i) = 0.2f;
    }
    // initialize result matrix
    for (int i = 0; i < 1024; i++) {
        for (int j = 0; j < 1024; j++) {
            result[i][j] = 0;
        }
    }

    for (int i = 0; i < 1024; i++) {
        for (int j = 0; j < 1024; j++) {
            for (int k = 0; k < 1024; k += 8) {
                // load
                va = _mm256_loadu_ps(matrix_a+(i*1024)+k); // matrix_a[i][k]
                vb = _mm256_loadu_ps(matrix_b+(j*1024)+k); // matrix_b[j][k]

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
                result[i][j] += _mm_cvtss_f32(vresult);
            }
        }
    }
    
    for (int i = 0; i < 1024; i++) {
        for (int j = 0; j < 1024; j++) {
            printf("%f ", result[i][j]);
        }
        printf("\n");
    }
    
    return 0;
}
