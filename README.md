# ugemm

public domain Simple, Minimalistic, Fast GEMM library

## How to build on macOS

```bash
$ make
```

## How to build on Linux

```bash
# cat /etc/yum.repos.d/rocm.repo 
[ROCm]
name=ROCm
#baseurl=http://repo.radeon.com/rocm/yum/2.2/
baseurl=http://repo.radeon.com/rocm/yum/4.0/
enabled=1
gpgcheck=0

# dnf install opencl-headers mesa-libOpenCL ocl-icd-devel
# dnf install rocm-clang-ocl rocm-opencl rocm-opencl-devel rocm-utils
$ gcc -O3 sgemm_ocl.c -o sgemm_ocl -lOpenCL -lm

$ make
```

- https://github.com/RadeonOpenCompute/ROCm
- https://www.amd.com/en/support/kb/release-notes/rn-amdgpu-unified-linux-20-10
- https://linuxreviews.org/Radeon_Open_Compute

## How to use

```bash
$ ./sgemm_ocl1
AMD Radeon HD 7800 Series (TAHITI, DRM 3.35.0, 5.4.6-berry, LLVM 9.0.0) (platform 0/1, device 0/2)
Maximum memory allocation size is 2576980377 bytes
>>> Done: took 0.032 seconds per run, 62.9 GFLOPS
0.000e+00/1.235e+20=0.000e+00. 0.000e+00 at [  0,  0]  -3.9929586175e+14 vs  -3.9929586175e+14 

$ ./sgemm_ocl2
AMD Radeon HD 7800 Series (TAHITI, DRM 3.35.0, 5.4.6-berry, LLVM 9.0.0) (platform 0/1, device 0/2)
Maximum memory allocation size is 2576980377 bytes
>>> Done: took 0.016 seconds per run, 122.3 GFLOPS
0.000e+00/1.235e+20=0.000e+00. 0.000e+00 at [  0,  0]  -3.9929586175e+14 vs  -3.9929586175e+14 

$ ./sgemm_ocl3
AMD Radeon HD 7800 Series (TAHITI, DRM 3.35.0, 5.4.6-berry, LLVM 9.0.0) (platform 0/1, device 0/2)
Maximum memory allocation size is 2576980377 bytes
>>> Done: took 0.018 seconds per run, 112.6 GFLOPS
0.000e+00/1.235e+20=0.000e+00. 0.000e+00 at [  0,  0]  -3.9929586175e+14 vs  -3.9929586175e+14 

$ ./sgemm_ocl4
AMD Radeon HD 7800 Series (TAHITI, DRM 3.35.0, 5.4.6-berry, LLVM 9.0.0) (platform 0/1, device 0/2)
Maximum memory allocation size is 2576980377 bytes
>>> Done: took 0.015 seconds per run, 131.8 GFLOPS
0.000e+00/1.235e+20=0.000e+00. 0.000e+00 at [  0,  0]  -3.9929586175e+14 vs  -3.9929586175e+14 

$ ./sgemm_ocl6
AMD Radeon HD 7800 Series (TAHITI, DRM 3.35.0, 5.4.6-berry, LLVM 9.0.0) (platform 0/1, device 0/2)
Maximum memory allocation size is 2576980377 bytes
>>> Done: took 0.012 seconds per run, 163.9 GFLOPS
0.000e+00/1.463e+20=0.000e+00. 0.000e+00 at [  0,  0]  -3.6711264766e+20 vs  -3.6711264766e+20 

$ ./sgemm-fast_ocl 
AMD Radeon HD 7800 Series (TAHITI, DRM 3.35.0, 5.4.6-berry, LLVM 9.0.0) (platform 0/1, device 0/2)
Maximum memory allocation size is 2576980377 bytes
>>> Done: took 0.012 seconds per run, 162.1 GFLOPS
0.000e+00/1.463e+20=0.000e+00. 0.000e+00 at [  0,  0]  -3.6711264766e+20 vs  -3.6711264766e+20 

$ FORCE_CPU=1 ./sgemm_ocl
pthread-Intel(R) Xeon(R) CPU E5-1650 v2 @ 3.50GHz (platform 0/2, device 0/1)
Maximum memory allocation size is 4294967296 bytes
>>> Done: took 0.108 seconds per run, 19.8 GFLOPS
0.000e+00/3.849e+21=0.000e+00. 0.000e+00 at [  0,  0]   2.3661284071e+18 vs   2.3661284071e+18 

$ ./sgemm_ocl -p 1
AMD Radeon HD 7800 Series (TAHITI, DRM 3.35.0, 5.4.6-berry, LLVM 9.0.0) (platform 1/2, device 0/2)
Maximum memory allocation size is 2576980377 bytes
>>> Done: took 0.015 seconds per run, 146.7 GFLOPS
0.000e+00/3.849e+21=0.000e+00. 0.000e+00 at [  0,  0]   2.3661284071e+18 vs   2.3661284071e+18 

```

## Reference

- [Tutorial: OpenCL SGEMM tuning for Kepler](https://cnugteren.github.io/tutorial/pages/page1.html)
- [GEMM: From Pure C to SSE Optimized Micro Kernels](http://apfel.mathematik.uni-ulm.de/~lehn/sghpc/gemm/)
- [GPU Powered BLAS for Browsers](https://github.com/waylonflinn/weblas)
- [いまどきのmatmul](http://int.main.jp/txt/matmul/)
- [SGEMM tester](https://github.com/gcp/sgemm)
- [DNN framework using im2col](https://github.com/hiroyam/dnn-im2col)
