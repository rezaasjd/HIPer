#ifndef KERNELS_HPP
#define KERNELS_HPP
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

template <int M, int K, int N>
__global__ void naive_mm(const float *A, const float *B, float *C);

template <int M, int K, int N, int BLOCK_SIZE>
__global__ void blocked_mm(const float *A, const float *B, float *C);

void rocwmma_gemm(const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    float alpha,
    float beta,
    hipStream_t stream);

#endif // KERNELS_HPP
