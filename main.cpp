#include <filesystem>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/driver_types.h>
#include <iostream>
#include <vector>
#include <random>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include "utils.hpp"


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

template <int M, int K, int N>
void mm(const float* A, const float* B, float* C)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float dot = 0.0f;
            for (int k = 0; k < K; ++k) {
                dot += A[i*K+k] * B[j+k*N];
            }
            C[i*M+j] = dot;
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc == 1) {
        std::cout << "Please provide number of iterations to be run.\n";
        return 1;
    }
    int cmdln_iter = atoi(argv[1]);
    constexpr int M = 2048;
    constexpr int K = 1024;
    constexpr int N = 2048;

    std::cout << std::left;
    HIP_CHECK(hipSetDevice(0));

    section("Populating buffers");
    std::vector<float> host_a(M*K);
    std::vector<float> host_b(K*N);

    std::random_device rd;
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    auto mat_a_filename = std::format("matrix_a_{}_{}.raw", M, K);
    auto mat_b_filename = std::format("matrix_b_{}_{}.raw", K, N);

    if (std::filesystem::exists(mat_a_filename)) {
        std::cout << "Reading matrix A from file " << mat_a_filename << std::endl;
        load_matrix(mat_a_filename, host_a.data());
    } else {
        for (size_t i = 0; i < M*K; ++i) {
            host_a[i] = dist(rd);
        }
        save_matrix(mat_a_filename, host_a.data(), M, K);
    }
    if (std::filesystem::exists(mat_b_filename)) {
        std::cout << "Reading matrix B from file " << mat_b_filename << std::endl;
        load_matrix(mat_b_filename, host_b.data());

    } else {
        for (size_t i = 0; i < K*N; ++i) {
            host_b[i] = dist(rd);
        }
        save_matrix(mat_a_filename, host_b.data(), K, N);
    }
    section("Allocating device buffers");
    float *dev_a, *dev_b, *dev_c;
    HIP_CHECK(hipMalloc(&dev_a, M*K*sizeof(float)));
    HIP_CHECK(hipMalloc(&dev_b, K*N*sizeof(float)));
    HIP_CHECK(hipMalloc(&dev_c, M*N*sizeof(float)));

    section("Copying over to device");
    HIP_CHECK(hipMemcpy(dev_a, const_cast<float*>(host_a.data()), M*K*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_b, const_cast<float*>(host_b.data()), M*K*sizeof(float), hipMemcpyHostToDevice));

    dim3 block_dims(16, 16);
    dim3 grid_dims(((M + block_dims.x - 1) / block_dims.x),
                   ((N + block_dims.y - 1) / block_dims.y));

    for (int iter = 0; iter < cmdln_iter; ++iter) {
        section(std::format("Kernel launch{}", iter));
        blocked_mm<M, K, N, 32><<<grid_dims, block_dims>>>
            (dev_a, dev_b, dev_c);
    }

    section("Copying results");
    std::vector<float> host_c(M*N);
    HIP_CHECK(hipMemcpy(host_c.data(), dev_c, M*N*sizeof(float), hipMemcpyDeviceToHost));

    section("CPU Kernel launch");
    std::vector<float> host_c_cpu(M*N);
    mm<M, K, N>(host_a.data(), host_b.data(), host_c_cpu.data());

    section("Comparison");
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (MATCH_ELEM(host_c[i*N+j], host_c_cpu[i*N+j], K)) {
                std::cout << "Error: mismatch at (" << i << ", " << j << "): " << host_c_cpu[i*N+j] << " != "
                    << host_c[i*N+j] << std::endl;
                return 1;
            }
        }
    }

    section("Success!");
    return 0;
}
