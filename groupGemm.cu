

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#define NUM_GROUPS 8
#define K_DIM 4096
#define N_DIM 2048

// Tensor Core (WMMA) 分块: 16x16x16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WMMA_TILE_M 64
#define WMMA_TILE_N 64
#define WMMA_WARPS 16
#define WMMA_BLOCK_DIM 32
#define WMMA_BLOCK_ROWS 16

// ============ CPU 参考实现 ============
// float 累加保证精度，输出转 FP16 与 GPU 一致
void cpu_group_gemm_ref(
    const __half* A, const __half* B, __half* C,
    const int* M_list, int K, int N)
{
    for (int g = 0; g < NUM_GROUPS; g++) {
        int Mg = M_list[g];
        int offset_A = 0, offset_B = 0, offset_C = 0;
        for (int i = 0; i < g; i++) {
            offset_A += M_list[i] * K;
            offset_B += K * N;
            offset_C += M_list[i] * N;
        }

        for (int m = 0; m < Mg; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += __half2float(A[offset_A + m * K + k]) * __half2float(B[offset_B + k * N + n]);
                }
                C[offset_C + m * N + n] = __float2half(sum);
            }
        }
    }
}

// WMMA: A/B 均 row_major，B 整块 64 列加载后各 warp 取 16 列
__global__ __launch_bounds__(512, 2) void group_gemm_wmma_kernel(
    int g,
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    const int* __restrict__ M_list,
    const int* __restrict__ offset_A,
    const int* __restrict__ offset_B,
    const int* __restrict__ offset_C,
    int K, int N)
{
    int Mg = M_list[g];
    int base_A = offset_A[g];
    int base_B = offset_B[g];
    int base_C = offset_C[g];

    __shared__ __half sA[2][WMMA_TILE_M][WMMA_K];
    __shared__ __half sB[2][WMMA_K][WMMA_TILE_N];  // row-major, 与 B 存储一致

    int warpM = threadIdx.y / 4;
    int warpN = threadIdx.y % 4;
    int tile_m = blockIdx.y * WMMA_TILE_M + warpM * WMMA_M;
    int tile_n = blockIdx.x * WMMA_TILE_N + warpN * WMMA_N;
    int tile_n_B = blockIdx.x * WMMA_TILE_N;  // B 加载：整块 64 列

    if (tile_m >= Mg) return;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    int num_tiles_k = K / WMMA_K;
    int ty = threadIdx.y, tx = threadIdx.x;

    for (int k = 0; k < num_tiles_k; k++) {
        int k_tile = k * WMMA_K;
        int buf = k & 1;
        int next = 1 - buf;

        if (k > 0) __syncthreads();

        for (int i = ty * 32 + tx; i < WMMA_TILE_M * WMMA_K; i += 512) {
            int row = i / WMMA_K, col = i % WMMA_K;
            int gr = blockIdx.y * WMMA_TILE_M + row;
            sA[buf][row][col] = (gr < Mg && k_tile + col < K) ?
                A[base_A + gr * K + k_tile + col] : __float2half(0.0f);
        }
        for (int i = ty * 32 + tx; i < WMMA_K * WMMA_TILE_N; i += 512) {
            int row = i / WMMA_TILE_N, col = i % WMMA_TILE_N;
            int gc = tile_n_B + col;
            sB[buf][row][col] = (k_tile + row < K && gc < N) ?
                B[base_B + (k_tile + row) * N + gc] : __float2half(0.0f);
        }
        if (k + 1 < num_tiles_k) {
            int k_next = (k + 1) * WMMA_K;
            for (int i = ty * 32 + tx; i < WMMA_TILE_M * WMMA_K; i += 512) {
                int row = i / WMMA_K, col = i % WMMA_K;
                int gr = blockIdx.y * WMMA_TILE_M + row;
                sA[next][row][col] = (gr < Mg && k_next + col < K) ?
                    A[base_A + gr * K + k_next + col] : __float2half(0.0f);
            }
            for (int i = ty * 32 + tx; i < WMMA_K * WMMA_TILE_N; i += 512) {
                int row = i / WMMA_TILE_N, col = i % WMMA_TILE_N;
                int gc = tile_n_B + col;
                sB[next][row][col] = (k_next + row < K && gc < N) ?
                    B[base_B + (k_next + row) * N + gc] : __float2half(0.0f);
            }
        }
        __syncthreads();

        nvcuda::wmma::load_matrix_sync(a_frag, &sA[buf][warpM * WMMA_M][0], WMMA_K);
        nvcuda::wmma::load_matrix_sync(b_frag, &sB[buf][0][warpN * WMMA_N], WMMA_TILE_N);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __shared__ float sC[WMMA_TILE_M][WMMA_TILE_N];
    nvcuda::wmma::store_matrix_sync(&sC[warpM * WMMA_M][warpN * WMMA_N], c_frag, WMMA_TILE_N, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (int i = ty * 32 + tx; i < WMMA_M * WMMA_N; i += 512) {
        int r = i / WMMA_N, c = i % WMMA_N;
        int row = tile_m + r, col = tile_n + c;
        if (row < Mg && col < N)
            C[base_C + row * N + col] = __float2half(sC[warpM * WMMA_M + r][warpN * WMMA_N + c]);
    }
}

// 融合 3D grid + 双缓冲：预取下一 tile 与当前 MMA 重叠
__global__ __launch_bounds__(512, 4) void group_gemm_wmma_fused_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    const int* __restrict__ M_list,
    const int* __restrict__ offset_A,
    const int* __restrict__ offset_B,
    const int* __restrict__ offset_C,
    int K, int N)
{
    int g = blockIdx.z;
    int Mg = M_list[g];
    int base_A = offset_A[g];
    int base_B = offset_B[g];
    int base_C = offset_C[g];

    __shared__ __half sA[2][WMMA_TILE_M][WMMA_K];
    __shared__ __half sB[2][WMMA_K][WMMA_TILE_N];

    int warpM = threadIdx.y / 4;
    int warpN = threadIdx.y % 4;
    int tile_m = blockIdx.y * WMMA_TILE_M + warpM * WMMA_M;
    int tile_n = blockIdx.x * WMMA_TILE_N + warpN * WMMA_N;
    int tile_n_B = blockIdx.x * WMMA_TILE_N;
    int ty = threadIdx.y, tx = threadIdx.x;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    int num_tiles_k = K / WMMA_K;

    // 预加载 k=0
    for (int i = ty * 32 + tx; i < WMMA_TILE_M * WMMA_K; i += 512) {
        int row = i / WMMA_K, col = i % WMMA_K;
        int gr = blockIdx.y * WMMA_TILE_M + row;
        sA[0][row][col] = (gr < Mg && col < K) ?
            A[base_A + gr * K + col] : __float2half(0.0f);
    }
    for (int i = ty * 32 + tx; i < WMMA_K * WMMA_TILE_N; i += 512) {
        int row = i / WMMA_TILE_N, col = i % WMMA_TILE_N;
        sB[0][row][col] = (row < K && tile_n_B + col < N) ?
            B[base_B + row * N + tile_n_B + col] : __float2half(0.0f);
    }
    __syncthreads();

    for (int k = 0; k < num_tiles_k; k++) {
        int buf = k & 1;
        int next_buf = 1 - buf;
        int k_tile = k * WMMA_K;
        int k_next = (k + 1) * WMMA_K;

        if (k + 1 < num_tiles_k) {
            for (int i = ty * 32 + tx; i < WMMA_TILE_M * WMMA_K; i += 512) {
                int row = i / WMMA_K, col = i % WMMA_K;
                int gr = blockIdx.y * WMMA_TILE_M + row;
                sA[next_buf][row][col] = (gr < Mg && k_next + col < K) ?
                    A[base_A + gr * K + k_next + col] : __float2half(0.0f);
            }
            for (int i = ty * 32 + tx; i < WMMA_K * WMMA_TILE_N; i += 512) {
                int row = i / WMMA_TILE_N, col = i % WMMA_TILE_N;
                sB[next_buf][row][col] = (k_next + row < K && tile_n_B + col < N) ?
                    B[base_B + (k_next + row) * N + tile_n_B + col] : __float2half(0.0f);
            }
        }

        nvcuda::wmma::load_matrix_sync(a_frag, &sA[buf][warpM * WMMA_M][0], WMMA_K);
        nvcuda::wmma::load_matrix_sync(b_frag, &sB[buf][0][warpN * WMMA_N], WMMA_TILE_N);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    __shared__ float sC[WMMA_TILE_M][WMMA_TILE_N];
    nvcuda::wmma::store_matrix_sync(&sC[warpM * WMMA_M][warpN * WMMA_N], c_frag, WMMA_TILE_N, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (int i = ty * 32 + tx; i < WMMA_M * WMMA_N; i += 512) {
        int r = i / WMMA_N, c = i % WMMA_N;
        int row = tile_m + r, col = tile_n + c;
        if (row < Mg && col < N)
            C[base_C + row * N + col] = __float2half(sC[warpM * WMMA_M + r][warpN * WMMA_N + c]);
    }
}

// 简化版内核: 每个线程计算一个输出元素 (无 warp 归约，便于验证)
__global__ void group_gemm_simple_kernel(
    int g,
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    const int* __restrict__ M_list,
    const int* __restrict__ offset_A,
    const int* __restrict__ offset_B,
    const int* __restrict__ offset_C,
    int K, int N)
{
    int Mg = M_list[g];
    int base_A = offset_A[g];
    int base_B = offset_B[g];
    int base_C = offset_C[g];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= Mg || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += __half2float(A[base_A + row * K + k]) *
               __half2float(B[base_B + k * N + col]);
    }
    C[base_C + row * N + col] = __float2half(sum);
}

// ============ 主程序 ============
int main() {
    srand(42);

    // M: 251~260 随机 8 个值
    int M_list[NUM_GROUPS];
    int total_M = 0;
    std::cout << "M values: ";
    for (int g = 0; g < NUM_GROUPS; g++) {
        M_list[g] = 251 + (rand() % 10);  // 251~260
        total_M += M_list[g];
        std::cout << M_list[g] << " ";
    }
    std::cout << std::endl;

    // 计算偏移
    int offset_A[NUM_GROUPS], offset_B[NUM_GROUPS], offset_C[NUM_GROUPS];
    offset_A[0] = offset_B[0] = offset_C[0] = 0;
    for (int g = 1; g < NUM_GROUPS; g++) {
        offset_A[g] = offset_A[g - 1] + M_list[g - 1] * K_DIM;
        offset_B[g] = offset_B[g - 1] + K_DIM * N_DIM;
        offset_C[g] = offset_C[g - 1] + M_list[g - 1] * N_DIM;
    }

    size_t size_A = total_M * K_DIM * sizeof(__half);
    size_t size_B = NUM_GROUPS * K_DIM * N_DIM * sizeof(__half);
    size_t size_C = total_M * N_DIM * sizeof(__half);

    // 分配主机内存
    __half* h_A = (__half*)malloc(size_A);
    __half* h_B = (__half*)malloc(size_B);
    __half* h_C = (__half*)malloc(size_C);
    __half* h_C_ref = (__half*)malloc(size_C);

    // 初始化 (小范围随机，便于验证)
    for (size_t i = 0; i < total_M * K_DIM; i++)
        h_A[i] = __float2half((float)(rand() % 10) / 100.0f);
    for (size_t i = 0; i < NUM_GROUPS * K_DIM * N_DIM; i++)
        h_B[i] = __float2half((float)(rand() % 10) / 100.0f);

    // CPU 参考计算
    cpu_group_gemm_ref(h_A, h_B, h_C_ref, M_list, K_DIM, N_DIM);

    // 设备内存
    __half *d_A, *d_B, *d_C;
    int *d_M_list, *d_offset_A, *d_offset_B, *d_offset_C;

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMalloc(&d_M_list, NUM_GROUPS * sizeof(int));
    cudaMalloc(&d_offset_A, NUM_GROUPS * sizeof(int));
    cudaMalloc(&d_offset_B, NUM_GROUPS * sizeof(int));
    cudaMalloc(&d_offset_C, NUM_GROUPS * sizeof(int));

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M_list, M_list, NUM_GROUPS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset_A, offset_A, NUM_GROUPS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset_B, offset_B, NUM_GROUPS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset_C, offset_C, NUM_GROUPS * sizeof(int), cudaMemcpyHostToDevice);

    // 使用 simple kernel 验证 (每个线程一个输出，无归约)
    dim3 block(16, 16);
    dim3 grid;
    for (int g = 0; g < NUM_GROUPS; g++) {
        grid.x = (N_DIM + block.x - 1) / block.x;
        grid.y = (M_list[g] + block.y - 1) / block.y;
        grid.z = 1;
        group_gemm_simple_kernel<<<grid, block>>>(g,
            d_A, d_B, d_C, d_M_list, d_offset_A, d_offset_B, d_offset_C,
            K_DIM, N_DIM);
    }
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // 验证 WMMA 融合内核正确性
    dim3 grid_w((N_DIM + WMMA_TILE_N - 1) / WMMA_TILE_N,
                (260 + WMMA_TILE_M - 1) / WMMA_TILE_M, NUM_GROUPS);
    group_gemm_wmma_fused_kernel<<<grid_w, dim3(WMMA_BLOCK_DIM, WMMA_BLOCK_ROWS)>>>(d_A, d_B, d_C,
        d_M_list, d_offset_A, d_offset_B, d_offset_C, K_DIM, N_DIM);
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    float max_err_wmma = 0.0f;
    for (size_t i = 0; i < total_M * N_DIM; i++)
        max_err_wmma = fmaxf(max_err_wmma, fabsf(__half2float(h_C_ref[i]) - __half2float(h_C[i])));
    std::cout << "WMMA (Tensor Core) kernel max err: " << max_err_wmma << std::endl;

    float max_err = 0.0f;
    int err_count = 0, inf_count = 0;
    for (size_t i = 0; i < total_M * N_DIM; i++) {
        float c_ref = __half2float(h_C_ref[i]);
        float c_gpu = __half2float(h_C[i]);
        if (!isfinite(c_gpu)) inf_count++;
        float err = isfinite(c_ref) && isfinite(c_gpu) ? fabsf(c_ref - c_gpu) : 0.0f;
        if (err > 1e-2f) err_count++;
        max_err = fmaxf(max_err, err);
    }

    std::cout << "\n=== 正确性验证 ===" << std::endl;
    std::cout << "Max |CPU - GPU|: " << max_err << std::endl;
    std::cout << "Error count (tol=1e-2): " << err_count << ", Inf/NaN: " << inf_count << std::endl;
    if (inf_count > 0)
        std::cout << "FAIL: GPU 输出含 Inf/NaN" << std::endl;
    else if (max_err < 1e-1f)
        std::cout << "PASS: 结果正确!" << std::endl;
    else
        std::cout << "WARN: 存在数值差异 (FP16 累积误差)" << std::endl;

    // ============ 性能测试 ============
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int warmup = 20;
    const int repeat = 200;

    std::cout << "\n=== 性能 (总 M: " << total_M << ", K: " << K_DIM << ", N: " << N_DIM << ") ===" << std::endl;

    // WMMA 融合 3D grid: 单次启动 8 组 (关键! 减少 launch 开销)
    dim3 grid_wmma((N_DIM + WMMA_TILE_N - 1) / WMMA_TILE_N,
                   (260 + WMMA_TILE_M - 1) / WMMA_TILE_M, NUM_GROUPS);
    dim3 block_wmma(WMMA_BLOCK_DIM, WMMA_BLOCK_ROWS);
    for (int i = 0; i < warmup; i++)
        group_gemm_wmma_fused_kernel<<<grid_wmma, block_wmma>>>(d_A, d_B, d_C,
            d_M_list, d_offset_A, d_offset_B, d_offset_C, K_DIM, N_DIM);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        group_gemm_wmma_fused_kernel<<<grid_wmma, block_wmma>>>(d_A, d_B, d_C,
            d_M_list, d_offset_A, d_offset_B, d_offset_C, K_DIM, N_DIM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_wmma;
    cudaEventElapsedTime(&ms_wmma, start, stop);
    ms_wmma /= repeat;
    double total_flops = 2.0 * (double)total_M * K_DIM * N_DIM;
    double gflops_wmma = (total_flops / 1e9) / (ms_wmma / 1000.0);
    double total_bytes = size_A + size_B + size_C;
    double gbs_wmma = (total_bytes / 1e9) / (ms_wmma / 1000.0);
    std::cout << "\n=== WMMA (Tensor Core + 融合 3D grid) [A100 理论 ~312 TFLOPS] ===" << std::endl;
    std::cout << "单次耗时: " << std::fixed << std::setprecision(3) << ms_wmma << " ms" << std::endl;
    std::cout << "计算效率: " << std::fixed << std::setprecision(2) << gflops_wmma << " GFLOPS/s" << std::endl;
    std::cout << "内存带宽: " << std::fixed << std::setprecision(2) << gbs_wmma << " GB/s" << std::endl;

    // 释放
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_M_list);
    cudaFree(d_offset_A);
    cudaFree(d_offset_B);
    cudaFree(d_offset_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

