
#include "kernels.hpp"
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma_impl.hpp>

using namespace rocwmma;

template <int M, int K, int N>
__global__ void naive_mm(const float *A, const float *B, float *C)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < M and j < N) {
        float dot = 0.0f;
        for (int k = 0; k < K; ++k) {
            dot += A[i*K+k] * B[k*N+j];
        }
        C[i*N+j] = dot;
    }
}

template __global__ void blocked_mm<2048, 1024, 2048, 32>(const float *A, const float *B, float *C);


template <int M, int K, int N, int BLOCK_SIZE>
__global__ void blocked_mm(const float *A, const float *B, float *C)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int j = BLOCK_SIZE * by + ty;
    const int i = BLOCK_SIZE * bx + tx;
    const int phases = (BLOCK_SIZE+K-1) / BLOCK_SIZE;

    __shared__ float _A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float _B[BLOCK_SIZE][BLOCK_SIZE];

    float thread_dot  = 0.0f;
    for (int phase = 0; phase < phases; ++phase) {
        if ((i < M) and ((phase*BLOCK_SIZE+tx) < N))
            _A[ty][tx] = A[(i)*K + phase * BLOCK_SIZE + tx];
        else
            _A[ty][tx] = 0.0f;

        if (((phase*BLOCK_SIZE+ty) < K) and (j < N))
            _B[ty][tx] = B[(phase*BLOCK_SIZE+ty)*N+j];
        else
            _B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            thread_dot += _A[ty][k] * _B[k][ty];
        }
        __syncthreads();
    }
    if (i < M and j < N)
        C[i*N+j] = thread_dot;
}


template<int BLOCK_SIZE = 256>
__global__ void rocwmma_gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    float alpha = 1.0f,
    float beta = 0.0f
) {
    // ROCwmma fragment dimensions
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    // Block and thread indices
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int warpRow = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;
    int warpCol = 0; // Single warp per block in this simple version

    // Calculate global matrix position
    int globalRow = blockRow * WMMA_M + warpRow * WMMA_M;
    int globalCol = blockCol * WMMA_N + warpCol * WMMA_N;

    // Bounds check
    if (globalRow >= M || globalCol >= N) return;

    // Declare ROCwmma fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, float, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, float, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Initialize accumulator
    fill_fragment(acc_frag, 0.0f);

    // Load initial C values if beta != 0
    if (beta != 0.0f) {
        load_matrix_sync(c_frag, C + globalRow * ldc + globalCol, ldc);
        // Scale by beta
        for (int i = 0; i < c_frag.num_elements; ++i) {
            acc_frag.x[i] += beta * c_frag.x[i];
        }
    }

    // Main computation loop over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        // Bounds check for K dimension
        if (k + WMMA_K > K) break;

        // Load A and B fragments
        load_matrix_sync(a_frag, A + globalRow * lda + k, lda);
        load_matrix_sync(b_frag, B + k * ldb + globalCol, ldb);

        // Perform matrix multiplication and accumulate
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Scale accumulator by alpha
    if (alpha != 1.0f) {
        for (int i = 0; i < acc_frag.num_elements; ++i) {
            acc_frag.x[i] *= alpha;
        }
    }

    // Store result
    store_matrix_sync(C + globalRow * ldc + globalCol, acc_frag, ldc);
}

// Host wrapper function
void rocwmma_gemm(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    float alpha = 1.0f,
    float beta = 0.0f,
    hipStream_t stream = 0
) {
    // ROCwmma works with 16x16 tiles
    constexpr int TILE_SIZE = 16;

    // Calculate grid dimensions
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(64, 1); // Adjust based on your GPU

    hipLaunchKernelGGL(
        rocwmma_gemm_kernel<>,
        grid, block, 0, stream,
        d_A, d_B, d_C, M, N, K, lda, ldb, ldc, alpha, beta
    );
}



