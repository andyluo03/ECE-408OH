// https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/

#include <chrono>
#include <iostream>

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

constexpr int kTileHeight = 32;
constexpr int kTileWidth  = 32;

__global__ 
void transpose_no_swizzle (const float* in, float* out, int M, int N) {
    __shared__ float tile[kTileHeight][kTileWidth];

    // Aligned w/ Input Tile.
    int blockInY = blockIdx.y * 32;
    int blockInX = blockIdx.x * 32;

    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    // 1. Load Data. 
    int global_load_stage_row = blockInX + warpId;
    int global_load_stage_col = blockInY + laneId;

    int shared_load_stage_row = warpId;
    int shared_load_stage_col = laneId;

    if (global_load_stage_row >= M || global_load_stage_col >= N) {
        tile[shared_load_stage_row][shared_load_stage_col] = 0;
    } else {
        tile[shared_load_stage_row][shared_load_stage_col] = in[global_load_stage_row * N + global_load_stage_col];
    }
    
    // 2. Synchronize Data.
    __syncthreads();

    // 3. Store Data.
    int global_write_stage_row = blockInY + warpId; // Transposed + Coalesced.
    int global_write_stage_col = blockInX + laneId;

    int shared_write_stage_row = laneId;
    int shared_write_stage_col = warpId;

    if (global_write_stage_row < N && global_write_stage_col < M) {
        out[global_write_stage_row * M + global_write_stage_col] = tile[shared_write_stage_row][shared_write_stage_col];
    }
}

void dispatch_transpose_no_swizzle (const float* in, float* out, int M, int N) {
    dim3 gridDim(ceil_div(M, kTileHeight), ceil_div(N, kTileWidth));
    dim3 blockDim(1024);
    transpose_no_swizzle<<<gridDim, blockDim>>>(in, out, M, N);
}

__global__ 
void transpose_pad_swizzle (const float* in, float* out, int M, int N) {
    __shared__ float tile[kTileHeight][kTileWidth + 1];

    // Aligned w/ Input Tile.
    int blockInY = blockIdx.y * 32;
    int blockInX = blockIdx.x * 32;

    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    // 1. Load Data. 
    int global_load_stage_row = blockInX + warpId;
    int global_load_stage_col = blockInY + laneId;

    int shared_load_stage_row = warpId;
    int shared_load_stage_col = laneId;

    if (global_load_stage_row >= M || global_load_stage_col >= N) {
        tile[shared_load_stage_row][shared_load_stage_col] = 0;
    } else {
        tile[shared_load_stage_row][shared_load_stage_col] = in[global_load_stage_row * N + global_load_stage_col];
    }
    
    // 2. Synchronize Data.
    __syncthreads();

    // 3. Store Data.
    int global_write_stage_row = blockInY + warpId; // Transposed + Coalesced.
    int global_write_stage_col = blockInX + laneId;

    int shared_write_stage_row = laneId;
    int shared_write_stage_col = warpId;

    if (global_write_stage_row < N && global_write_stage_col < M) {
        out[global_write_stage_row * M + global_write_stage_col] = tile[shared_write_stage_row][shared_write_stage_col];
    }
}

void dispatch_transpose_pad_swizzle (const float* in, float* out, int M, int N) {
    dim3 gridDim(ceil_div(M, kTileHeight), ceil_div(N, kTileWidth));
    dim3 blockDim(1024);
    transpose_pad_swizzle<<<gridDim, blockDim>>>(in, out, M, N);
}

__global__ 
void transpose_xor_swizzle (float* in, float* out) {

}

void dispatch_transpose_xor_swizzle (const float* in, float* out, int M, int N) {
    dim3 gridDim(ceil_div(M, kTileHeight), ceil_div(N, kTileWidth));
    dim3 blockDim(1024);
    transpose_pad_swizzle<<<gridDim, blockDim>>>(in, out, M, N);
}

int main () {
    int M = 8192;
    int N = 8192;

    float* h_in  = (float*) malloc(M * N * sizeof(float));
    for (int i = 0; i < M * N; i++) {
        h_in[i] = (float) rand() / (float) rand();
    }
    float* h_out  = (float*) malloc(M * N * sizeof(float));

    float* in;
    float* out_scratch;
    cudaMalloc(&in, M * N * sizeof(float));
    cudaMalloc(&out_scratch, M * N * sizeof(float));
    cudaMemcpy(in, h_in, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // 1.
    std::cout << "[ RUN: Naive Transpose]" << std::endl;

    auto start_naive = std::chrono::high_resolution_clock::now();

    dispatch_transpose_no_swizzle(in, out_scratch, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, in, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto end_naive = std::chrono::high_resolution_clock::now();

    std::cout << "Runtime: " << (std::chrono::duration_cast<std::chrono::nanoseconds>(end_naive - start_naive)).count() << " (ns)" << std::endl; 

    std::cout << "[ END: Naive Transpose]" << std::endl;

    // 2.
    std::cout << "[ RUN: Padded Transpose]" << std::endl;

    auto start_pad = std::chrono::high_resolution_clock::now();

    dispatch_transpose_pad_swizzle(in, out_scratch, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, in, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto end_pad = std::chrono::high_resolution_clock::now();

    std::cout << "Runtime: " << (std::chrono::duration_cast<std::chrono::nanoseconds>(end_pad - start_pad)).count() << " (ns)" << std::endl; 

    std::cout << "[ END: Padded Transpose]" << std::endl;

    // 3.
    std::cout << "[ RUN: XOR Transpose]" << std::endl;

    auto start_xor = std::chrono::high_resolution_clock::now();

    dispatch_transpose_xor_swizzle(in, out_scratch, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, in, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto end_xor = std::chrono::high_resolution_clock::now();

    std::cout << "Runtime: " << (std::chrono::duration_cast<std::chrono::nanoseconds>(end_xor - start_xor)).count() << " (ns)" << std::endl; 

    std::cout << "[ END: XOR Transpose]" << std::endl;


    std::cout << h_out[0] << std::endl; // Prevent Optimization.
}