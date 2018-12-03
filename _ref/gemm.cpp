#include <chrono>
#include <string>
#include <iostream>
#include <vector>

#include <immintrin.h>
//#include <cblas.h>

#ifdef USING_MKL
#include <mkl_service.h>
#endif

class BenchCb
{
public:
    BenchCb(const char* tag){
        _tag = tag;
        start_time = std::chrono::steady_clock::now();
        std::cout << _tag << ", testing..." << std::endl;
    }
    BenchCb(const char* tag, const int64_t float_ops){
        _float_ops = float_ops;
        _tag = tag;
        start_time = std::chrono::steady_clock::now();
        std::cout << _tag << ", testing..." << std::endl;
    }
    ~BenchCb(){
        auto stop_time = std::chrono::steady_clock::now();
        const float cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
        if (_float_ops > 0)
        {
            const float gflops = (_float_ops / (cost_time/1000.0f)) / (1024.0f*1024.0f*1024.0f);
            std::cout << _tag << ", cost time: " << cost_time << " ms. GLOPS: " << gflops << std::endl;
        }
        else
        {
            std::cout << _tag << ", cost time: " << cost_time << " ms. " << std::endl;
        }
    }
private:
    std::string _tag;
    int64_t _float_ops = 0;
    decltype(std::chrono::steady_clock::now()) start_time;
};

static void reset_state(std::vector<float>& matA, std::vector<float>& matB, std::vector<float>& matC)
{
    for (int i = 0; i < matA.size();i++)
    {
        matA[i] = (i % 8);
    }
    for (int i = 0; i < matB.size(); i++)
    {
        matB[i] = (i % 8);
    }
    for (int i = 0; i < matC.size(); i++)
    {
        matC[i] = 0.0f;
    }
}
static void assert_equals(const std::vector<float>& matLhs, const std::vector<float>& matRhs)
{
    if (matLhs.size() != matRhs.size())
    {
        std::cout << "dim is not equals!" << std::endl;
        exit(0);
    }
    for (int i = 0; i < matLhs.size(); i++)
    {
        if (matLhs[i] != matRhs[i])
        {
            std::cout << "lhs is not equals with rhs! idx: "<< i << ", lhs:" << matLhs[i] <<", rhs: "<<matRhs[i] << std::endl;
            exit(0);
        }
    }
}
void bench_gemm(const int _C, const int _M, const int _N, const int _K)
{
#if defined(USING_OPENBLAS)
    openblas_set_num_threads(1);
#elif defined(USING_MKL)
    mkl_set_num_threads(1);
#endif

    std::vector<float> matA(_M*_K);
    std::vector<float> matB(_K*_N);
    std::vector<float> matC(_M*_N);
    const int64_t float_ops = (int64_t)(_M)*_N*_K*_C;

    std::vector<float> baseline;

    {       
        BenchCb cb("baseline", float_ops);
        [&](){
            for (int c = 0; c < _C; c++)
            {
                reset_state(matA, matB, matC);
                for (int i = 0; i < _M; i++)
                {
                    for (int j = 0; j < _N; j++)
                    {
                        for (int k = 0; k < _K; k++)
                        {
                            matC[i*_N + j] += matA[i*_K + k] * matB[k*_N + j];
                        }
                    }
                }
            }
        }();
        baseline = matC;
    }

    /*{
        reset_state(matA, matB, matC);
        BenchCb cb("gemm_cb", float_ops);
        [&](){
            for (int c = 0; c < _C; c++)
            {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _M, _N, _K, 1.0f, &matA[0], _K, &matB[0], _N, 0.0f, &matC[0], _N);
            }
        }();
        assert_equals(baseline, matC);
    }*/

    {
        BenchCb cb("gemm_v1", float_ops);
        auto core_cb = [](const float* rowA, const float* colB, float* matC, const int _M, const int _N, const int _K){
            for (int k = 0; k < _K; k++)
            {
                //4
                matC[0] += rowA[k] * colB[k*_N + 0];
                matC[1] += rowA[k] * colB[k*_N + 1];
                matC[2] += rowA[k] * colB[k*_N + 2];
                matC[3] += rowA[k] * colB[k*_N + 3];
            }
        };
        [&](){
            for (int c = 0; c < _C; c++)
            {
                reset_state(matA, matB, matC);
                for (int j = 0; j < _N; j += 4)
                {
                    for (int i = 0; i < _M; i++)
                    {
                        core_cb(&matA[i*_K], &matB[j], &matC[i * _N + j], _M, _N, _K);
                    }
                }
            }
        }();
        assert_equals(baseline, matC);
    }

    {
        BenchCb cb("gemm_v2", float_ops);
        auto core_cb = [](const float* rowA, const float* colB, float* matC, const int _M, const int _N, const int _K){
            register float reg_a = 0.0f;
            register float reg_c0 = 0.0f;
            register float reg_c1 = 0.0f;
            register float reg_c2 = 0.0f;
            register float reg_c3 = 0.0f;
            for (int k = 0; k < _K; k++)
            {
                reg_a = rowA[k];
                reg_c0 += reg_a * colB[k*_N + 0];
                reg_c1 += reg_a * colB[k*_N + 1];
                reg_c2 += reg_a * colB[k*_N + 2];
                reg_c3 += reg_a * colB[k*_N + 3];
            }
            matC[0] = reg_c0;
            matC[1] = reg_c1;
            matC[2] = reg_c2;
            matC[3] = reg_c3;
        };
        [&](){
            for (int c = 0; c < _C; c++)
            {
                reset_state(matA, matB, matC);
                for (int j = 0; j < _N; j += 4)
                {
                    for (int i = 0; i < _M; i++)
                    {
                        core_cb(&matA[i*_K], &matB[j], &matC[i * _N + j], _M, _N, _K);
                    }
                }
            }
        }();
        assert_equals(baseline, matC);
    }

    {
        BenchCb cb("gemm_v3", float_ops);
        auto core_cb = [](const float* rowA, const float* colB, float* matC, const int _M, const int _N, const int _K){
            register float reg_a = 0.0f;
            register float reg_c0 = 0.0f;
            register float reg_c1 = 0.0f;
            register float reg_c2 = 0.0f;
            register float reg_c3 = 0.0f;
            register const float* seg_b = colB;
            for (int k = 0; k < _K; k++)
            {
                reg_a = rowA[k];    
                
                reg_c0 += reg_a * seg_b[0];
                reg_c1 += reg_a * seg_b[1];
                reg_c2 += reg_a * seg_b[2];
                reg_c3 += reg_a * seg_b[3];

                seg_b += _N;                
            }
            matC[0] = reg_c0;
            matC[1] = reg_c1;
            matC[2] = reg_c2;
            matC[3] = reg_c3;
        };
        [&](){
            for (int c = 0; c < _C; c++)
            {
                reset_state(matA, matB, matC);
                for (int j = 0; j < _N; j += 4)
                {
                    for (int i = 0; i < _M; i++)
                    {
                        core_cb(&matA[i*_K], &matB[j], &matC[i * _N + j], _M, _N, _K);
                    }
                }
            }
        }();
        assert_equals(baseline, matC);
    }

    {
        BenchCb cb("gemm_v4", float_ops);
        auto core_cb = [](const float* rowA, const float* colB, float* matC, const int _M, const int _N, const int _K){
            register float reg_a = 0.0f;
            //row 0
            register float reg_c00 = 0.0f;
            register float reg_c01 = 0.0f;
            register float reg_c02 = 0.0f;
            register float reg_c03 = 0.0f;
            //row 1
            register float reg_c10 = 0.0f;
            register float reg_c11 = 0.0f;
            register float reg_c12 = 0.0f;
            register float reg_c13 = 0.0f;
            //row 2
            register float reg_c20 = 0.0f;
            register float reg_c21 = 0.0f;
            register float reg_c22 = 0.0f;
            register float reg_c23 = 0.0f;
            //row 3
            register float reg_c30 = 0.0f;
            register float reg_c31 = 0.0f;
            register float reg_c32 = 0.0f;
            register float reg_c33 = 0.0f;

            register const float* seg_b = colB;
            for (int k = 0; k < _K; k++)
            {
                //row 0
                reg_a = rowA[k + 0 * _K];
                reg_c00 += reg_a * seg_b[0];
                reg_c01 += reg_a * seg_b[1];
                reg_c02 += reg_a * seg_b[2];
                reg_c03 += reg_a * seg_b[3];                

                //row 1
                reg_a = rowA[k + 1 * _K];
                reg_c10 += reg_a * seg_b[0];
                reg_c11 += reg_a * seg_b[1];
                reg_c12 += reg_a * seg_b[2];
                reg_c13 += reg_a * seg_b[3];

                //row 2
                reg_a = rowA[k + 2 * _K];
                reg_c20 += reg_a * seg_b[0];
                reg_c21 += reg_a * seg_b[1];
                reg_c22 += reg_a * seg_b[2];
                reg_c23 += reg_a * seg_b[3];

                //row 3
                reg_a = rowA[k + 3 * _K];
                reg_c30 += reg_a * seg_b[0];
                reg_c31 += reg_a * seg_b[1];
                reg_c32 += reg_a * seg_b[2];
                reg_c33 += reg_a * seg_b[3];

                seg_b += _N;
            }
            //row 0
            matC[0 * _N + 0] = reg_c00;
            matC[0 * _N + 1] = reg_c01;
            matC[0 * _N + 2] = reg_c02;
            matC[0 * _N + 3] = reg_c03;
            //row 1
            matC[1 * _N + 0] = reg_c10;
            matC[1 * _N + 1] = reg_c11;
            matC[1 * _N + 2] = reg_c12;
            matC[1 * _N + 3] = reg_c13;
            //row 2
            matC[2 * _N + 0] = reg_c20;
            matC[2 * _N + 1] = reg_c21;
            matC[2 * _N + 2] = reg_c22;
            matC[2 * _N + 3] = reg_c23;
            //row 3
            matC[3 * _N + 0] = reg_c30;
            matC[3 * _N + 1] = reg_c31;
            matC[3 * _N + 2] = reg_c32;
            matC[3 * _N + 3] = reg_c33;
        };
        [&](){
            for (int c = 0; c < _C; c++)
            {
                reset_state(matA, matB, matC);
                for (int j = 0; j < _N; j += 4)
                {
                    for (int i = 0; i < _M; i += 4)
                    {
                        core_cb(&matA[i*_K], &matB[j], &matC[i * _N + j], _M, _N, _K);
                    }
                }
            }
        }();
        assert_equals(baseline, matC);
    }

    {
        BenchCb cb("gemm_v5", float_ops);
        auto core_cb = [](const float* rowA, const float* colB, float* matC, const int _M, const int _N, const int _K){
            __m128 reg_a = _mm_set1_ps(0.0f);
            __m128 reg_b = _mm_set1_ps(0.0f);
            //row 0
            __m128 reg_c00 = _mm_set1_ps(0.0f);
            //row 1
            __m128 reg_c10 = _mm_set1_ps(0.0f);
            //row 2
            __m128 reg_c20 = _mm_set1_ps(0.0f);
            //row 3
            __m128 reg_c30 = _mm_set1_ps(0.0f);

            const float* ptr_b = colB;
            for (int k = 0; k < _K; k++)
            {
                reg_b = _mm_loadu_ps(ptr_b);

                //row 0
                reg_a = _mm_set1_ps(rowA[k + 0 * _K]);              
                reg_c00 = _mm_add_ps(reg_c00, _mm_mul_ps(reg_a, reg_b));

                //row 1
                reg_a = _mm_set1_ps(rowA[k + 1 * _K]);
                reg_c10 = _mm_add_ps(reg_c10, _mm_mul_ps(reg_a, reg_b));

                //row 2
                reg_a = _mm_set1_ps(rowA[k + 2 * _K]);
                reg_c20 = _mm_add_ps(reg_c20, _mm_mul_ps(reg_a, reg_b));

                //row 3
                reg_a = _mm_set1_ps(rowA[k + 3 * _K]);
                reg_c30 = _mm_add_ps(reg_c30, _mm_mul_ps(reg_a, reg_b));

                ptr_b += _N;
            }
            //row 0
            _mm_storeu_ps(matC + 0 * _N, reg_c00);
            //row 1
            _mm_storeu_ps(matC + 1 * _N, reg_c10);
            //row 2
            _mm_storeu_ps(matC + 2 * _N, reg_c20);
            //row 3
            _mm_storeu_ps(matC + 3 * _N, reg_c30);
        };
        [&](){
            for (int c = 0; c < _C; c++)
            {
                reset_state(matA, matB, matC);
                for (int j = 0; j < _N; j += 4)
                {
                    for (int i = 0; i < _M; i += 4)
                    {
                        core_cb(&matA[i*_K], &matB[j], &matC[i * _N + j], _M, _N, _K);
                    }
                }
            }
        }();
        assert_equals(baseline, matC);
    }

    {
        BenchCb cb("gemm_v6", float_ops);
        auto core_cb = [](const float* rowA, const float* colB, float* matC, const int _M, const int _N, const int _K){
            __m256 reg_a = _mm256_set1_ps(0.0f);
            __m256 reg_b = _mm256_set1_ps(0.0f);
            //row 0
            __m256 reg_c00 = _mm256_set1_ps(0.0f);
            //row 1
            __m256 reg_c10 = _mm256_set1_ps(0.0f);
            //row 2
            __m256 reg_c20 = _mm256_set1_ps(0.0f);
            //row 3
            __m256 reg_c30 = _mm256_set1_ps(0.0f);
            //row 4
            __m256 reg_c40 = _mm256_set1_ps(0.0f);
            //row 5
            __m256 reg_c50 = _mm256_set1_ps(0.0f);
            //row 6
            __m256 reg_c60 = _mm256_set1_ps(0.0f);
            //row 7
            __m256 reg_c70 = _mm256_set1_ps(0.0f);

            const float* ptr_b = colB;
            for (int k = 0; k < _K; k++)
            {
                reg_b = _mm256_loadu_ps(ptr_b);

                //row 0
                reg_a = _mm256_set1_ps(rowA[k + 0 * _K]);
                reg_c00 = _mm256_add_ps(reg_c00, _mm256_mul_ps(reg_a, reg_b));

                //row 1
                reg_a = _mm256_set1_ps(rowA[k + 1 * _K]);
                reg_c10 = _mm256_add_ps(reg_c10, _mm256_mul_ps(reg_a, reg_b));

                //row 2
                reg_a = _mm256_set1_ps(rowA[k + 2 * _K]);
                reg_c20 = _mm256_add_ps(reg_c20, _mm256_mul_ps(reg_a, reg_b));

                //row 3
                reg_a = _mm256_set1_ps(rowA[k + 3 * _K]);
                reg_c30 = _mm256_add_ps(reg_c30, _mm256_mul_ps(reg_a, reg_b));

                //row 4
                reg_a = _mm256_set1_ps(rowA[k + 4 * _K]);
                reg_c40 = _mm256_add_ps(reg_c40, _mm256_mul_ps(reg_a, reg_b));

                //row 5
                reg_a = _mm256_set1_ps(rowA[k + 5 * _K]);
                reg_c50 = _mm256_add_ps(reg_c50, _mm256_mul_ps(reg_a, reg_b));

                //row 6
                reg_a = _mm256_set1_ps(rowA[k + 6 * _K]);
                reg_c60 = _mm256_add_ps(reg_c60, _mm256_mul_ps(reg_a, reg_b));

                //row 7
                reg_a = _mm256_set1_ps(rowA[k + 7 * _K]);
                reg_c70 = _mm256_add_ps(reg_c70, _mm256_mul_ps(reg_a, reg_b));

                ptr_b += _N;
            }
            //row 0
            _mm256_storeu_ps(matC + 0 * _N, reg_c00);
            //row 1
            _mm256_storeu_ps(matC + 1 * _N, reg_c10);
            //row 2
            _mm256_storeu_ps(matC + 2 * _N, reg_c20);
            //row 3
            _mm256_storeu_ps(matC + 3 * _N, reg_c30);
            //row 0
            _mm256_storeu_ps(matC + 4 * _N, reg_c40);
            //row 1
            _mm256_storeu_ps(matC + 5 * _N, reg_c50);
            //row 2
            _mm256_storeu_ps(matC + 6 * _N, reg_c60);
            //row 3
            _mm256_storeu_ps(matC + 7 * _N, reg_c70);
        };
        [&](){
            //TODO: package
            for (int c = 0; c < _C; c++)
            {
                reset_state(matA, matB, matC);
                for (int j = 0; j < _N; j += 8)
                {
                    for (int i = 0; i < _M; i += 8)
                    {
                        core_cb(&matA[i*_K], &matB[j], &matC[i * _N + j], _M, _N, _K);
                    }
                }
            }
        }();
        assert_equals(baseline, matC);
    }

    {
        BenchCb cb("gemm_v7", float_ops);
        auto core_cb = [](const float* rowA, const float* rowB, float* matC, const int _M, const int _N, const int _K){
            __m256 reg_a = _mm256_set1_ps(0.0f);
            __m256 reg_b = _mm256_set1_ps(0.0f);
            //row 0
            __m256 reg_c00 = _mm256_set1_ps(0.0f);
            //row 1
            __m256 reg_c10 = _mm256_set1_ps(0.0f);
            //row 2
            __m256 reg_c20 = _mm256_set1_ps(0.0f);
            //row 3
            __m256 reg_c30 = _mm256_set1_ps(0.0f);
            //row 4
            __m256 reg_c40 = _mm256_set1_ps(0.0f);
            //row 5
            __m256 reg_c50 = _mm256_set1_ps(0.0f);
            //row 6
            __m256 reg_c60 = _mm256_set1_ps(0.0f);
            //row 7
            __m256 reg_c70 = _mm256_set1_ps(0.0f);

            const float* ptr_b = rowB;
            for (int k = 0; k < _K; k++)
            {
                reg_b = _mm256_loadu_ps(ptr_b);

                //row 0
                reg_a = _mm256_set1_ps(rowA[k + 0 * _K]);
                reg_c00 = _mm256_add_ps(reg_c00, _mm256_mul_ps(reg_a, reg_b));

                //row 1
                reg_a = _mm256_set1_ps(rowA[k + 1 * _K]);
                reg_c10 = _mm256_add_ps(reg_c10, _mm256_mul_ps(reg_a, reg_b));

                //row 2
                reg_a = _mm256_set1_ps(rowA[k + 2 * _K]);
                reg_c20 = _mm256_add_ps(reg_c20, _mm256_mul_ps(reg_a, reg_b));

                //row 3
                reg_a = _mm256_set1_ps(rowA[k + 3 * _K]);
                reg_c30 = _mm256_add_ps(reg_c30, _mm256_mul_ps(reg_a, reg_b));

                //row 4
                reg_a = _mm256_set1_ps(rowA[k + 4 * _K]);
                reg_c40 = _mm256_add_ps(reg_c40, _mm256_mul_ps(reg_a, reg_b));

                //row 5
                reg_a = _mm256_set1_ps(rowA[k + 5 * _K]);
                reg_c50 = _mm256_add_ps(reg_c50, _mm256_mul_ps(reg_a, reg_b));

                //row 6
                reg_a = _mm256_set1_ps(rowA[k + 6 * _K]);
                reg_c60 = _mm256_add_ps(reg_c60, _mm256_mul_ps(reg_a, reg_b));

                //row 7
                reg_a = _mm256_set1_ps(rowA[k + 7 * _K]);
                reg_c70 = _mm256_add_ps(reg_c70, _mm256_mul_ps(reg_a, reg_b));

                ptr_b += 8;
            }
            //row 0
            _mm256_storeu_ps(matC + 0 * _N, reg_c00);
            //row 1
            _mm256_storeu_ps(matC + 1 * _N, reg_c10);
            //row 2
            _mm256_storeu_ps(matC + 2 * _N, reg_c20);
            //row 3
            _mm256_storeu_ps(matC + 3 * _N, reg_c30);
            //row 0
            _mm256_storeu_ps(matC + 4 * _N, reg_c40);
            //row 1
            _mm256_storeu_ps(matC + 5 * _N, reg_c50);
            //row 2
            _mm256_storeu_ps(matC + 6 * _N, reg_c60);
            //row 3
            _mm256_storeu_ps(matC + 7 * _N, reg_c70);
        };
        [&](){
            for (int c = 0; c < _C; c++)
            {
                reset_state(matA, matB, matC);
                std::vector<float> packMatB(matB.size());
                {
                    float* dst_unit = &packMatB[0];
                    for (int j = 0; j < _N; j += 8)
                    {
                        for (int k = 0; k < _K; k++)
                        {
                            const float* src_unit = &matB[k*_N + j];
                            _mm256_storeu_ps(dst_unit, _mm256_loadu_ps(src_unit));
                            dst_unit += 8;
                        }
                    }
                }
                for (int j = 0; j < _N; j += 8)
                {
                    for (int i = 0; i < _M; i += 8)
                    {
                        core_cb(&matA[i*_K], &packMatB[j*_K], &matC[i * _N + j], _M, _N, _K);
                    }
                }
            }
        }();
        assert_equals(baseline, matC);
    }

    {
        BenchCb cb("gemm_v8", float_ops);
        auto core_cb = [](const float* rowA, const float* rowB, float* matC, const int _M, const int _N, const int _K){
            __m256 reg_a = _mm256_set1_ps(0.0f);
            __m256 reg_b = _mm256_set1_ps(0.0f);
            //row 0
            __m256 reg_c00 = _mm256_set1_ps(0.0f);
            //row 1
            __m256 reg_c10 = _mm256_set1_ps(0.0f);
            //row 2
            __m256 reg_c20 = _mm256_set1_ps(0.0f);
            //row 3
            __m256 reg_c30 = _mm256_set1_ps(0.0f);
            //row 4
            __m256 reg_c40 = _mm256_set1_ps(0.0f);
            //row 5
            __m256 reg_c50 = _mm256_set1_ps(0.0f);
            //row 6
            __m256 reg_c60 = _mm256_set1_ps(0.0f);
            //row 7
            __m256 reg_c70 = _mm256_set1_ps(0.0f);

            //FMA3: _mm256_fmadd_ps
            //FMA4: _mm256_macc_ps
            const float* ptr_a = rowA;
            const float* ptr_b = rowB;
            for (int k = 0; k < _K; k++)
            {
                reg_b = _mm256_loadu_ps(ptr_b);

                //row 0
                reg_a = _mm256_set1_ps(ptr_a[0]);
                reg_c00 = _mm256_add_ps(reg_c00, _mm256_mul_ps(reg_a, reg_b));

                //row 1
                reg_a = _mm256_set1_ps(ptr_a[1]);
                reg_c10 = _mm256_add_ps(reg_c10, _mm256_mul_ps(reg_a, reg_b));

                //row 2
                reg_a = _mm256_set1_ps(ptr_a[2]);
                reg_c20 = _mm256_add_ps(reg_c20, _mm256_mul_ps(reg_a, reg_b));

                //row 3
                reg_a = _mm256_set1_ps(ptr_a[3]);
                reg_c30 = _mm256_add_ps(reg_c30, _mm256_mul_ps(reg_a, reg_b));

                //row 4
                reg_a = _mm256_set1_ps(ptr_a[4]);
                reg_c40 = _mm256_add_ps(reg_c40, _mm256_mul_ps(reg_a, reg_b));

                //row 5
                reg_a = _mm256_set1_ps(ptr_a[5]);
                reg_c50 = _mm256_add_ps(reg_c50, _mm256_mul_ps(reg_a, reg_b));

                //row 6
                reg_a = _mm256_set1_ps(ptr_a[6]);
                reg_c60 = _mm256_add_ps(reg_c60, _mm256_mul_ps(reg_a, reg_b));

                //row 7
                reg_a = _mm256_set1_ps(ptr_a[7]);
                reg_c70 = _mm256_add_ps(reg_c70, _mm256_mul_ps(reg_a, reg_b));

                ptr_a += 8;
                ptr_b += 8;
            }
            //row 0
            _mm256_storeu_ps(matC + 0 * _N, reg_c00);
            //row 1
            _mm256_storeu_ps(matC + 1 * _N, reg_c10);
            //row 2
            _mm256_storeu_ps(matC + 2 * _N, reg_c20);
            //row 3
            _mm256_storeu_ps(matC + 3 * _N, reg_c30);
            //row 0
            _mm256_storeu_ps(matC + 4 * _N, reg_c40);
            //row 1
            _mm256_storeu_ps(matC + 5 * _N, reg_c50);
            //row 2
            _mm256_storeu_ps(matC + 6 * _N, reg_c60);
            //row 3
            _mm256_storeu_ps(matC + 7 * _N, reg_c70);
        };
        [&](){
            for (int c = 0; c < _C; c++)
            {
                reset_state(matA, matB, matC);
                std::vector<float> packMatA(matA.size());
                {
                    float* dst_unit = &packMatA[0];
                    for (int i = 0; i < _M; i += 8)
                    {
                        for (int k = 0; k < _K; k++)
                        {
                            const float* src_unit = &matA[i*_K + k];                            
                            const __m256 src_data = _mm256_set_ps(src_unit[0 * _K], src_unit[1 * _K], src_unit[2 * _K], src_unit[3 * _K],
                                src_unit[4 * _K], src_unit[5 * _K], src_unit[6 * _K], src_unit[7 * _K]);
                            _mm256_storeu_ps(dst_unit, src_data);
                            dst_unit += 8;
                        }
                    }
                }
                std::vector<float> packMatB(matB.size());
                {
                    float* dst_unit = &packMatB[0];
                    for (int j = 0; j < _N; j += 8)
                    {
                        for (int k = 0; k < _K; k++)
                        {
                            const float* src_unit = &matB[k*_N + j];
                            _mm256_storeu_ps(dst_unit, _mm256_loadu_ps(src_unit));
                            dst_unit += 8;
                        }
                    }
                }
                for (int j = 0; j < _N; j += 8)
                {
                    for (int i = 0; i < _M; i += 8)
                    {
                        core_cb(&packMatA[j*_K], &packMatB[j*_K], &matC[i * _N + j], _M, _N, _K);
                    }
                }
            }
        }();
        assert_equals(baseline, matC);
    }
}

int main(int, char*[])
{
    for (int i = 1; i <= 32; i+=4)
    {       
        const int _M = i * 32;
        const int _N = i * 32;
        const int _K = i * 32;
        std::cout << "_M: " << _M << ", _N: " << _N << ", _K: " << _K << std::endl;
        bench_gemm(1,_M, _N, _K);
        std::cout << "\n\n" << std::endl;
    }
    return 0;
}
