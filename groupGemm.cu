
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#define LDG(p) __ldg(p)

#define NUM_GROUPS 8
#define K_DIM 4096
#define N_DIM 2048

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
// 小 M 优化: 32x32 tile 增加 block 数 (260/32=9 vs 260/64=5)
#define WMMA_TILE_M 32
#define WMMA_TILE_N 32
#define WMMA_BLOCK_DIM 32
#define WMMA_BLOCK_ROWS 4   // 4 warps (2x2) for 32x32

#define SA_LD 16
#define SB_LD 16
#define SC_LD 40

// ============ CPU 参考实现 (验证正确性) ============
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
                    sum += __half2float(A[offset_A + m * K + k]) * __half2float(B[offset_B + n * K + k]);
                }
                C[offset_C + m * N + n] = __float2half(sum);
            }
        }
    }
}

// ============ 手写 WMMA 融合 kernel (3D grid, 双缓冲, 32x32 tile 适配小 M) ============
__global__ __launch_bounds__(128, 8) void group_gemm_wmma_fused_kernel(
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

    if (blockIdx.y * WMMA_TILE_M >= Mg) return;

    __shared__ __half sA[2][WMMA_TILE_M][SA_LD];
    __shared__ __half sB[2][WMMA_TILE_N][SB_LD];

    int warpM = threadIdx.y / 2;  // 2x2 warps for 32x32 tile
    int warpN = threadIdx.y % 2;
    int tile_m = blockIdx.y * WMMA_TILE_M + warpM * WMMA_M;
    int tile_n = blockIdx.x * WMMA_TILE_N + warpN * WMMA_N;
    int tile_n_B = blockIdx.x * WMMA_TILE_N;
    int ty = threadIdx.y, tx = threadIdx.x;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    int num_tiles_k = K / WMMA_K;

#pragma unroll 4
    for (int i = ty * 32 + tx; i < WMMA_TILE_M * SA_LD; i += 128) {
        int row = i / SA_LD, col = i % SA_LD;
        int gr = blockIdx.y * WMMA_TILE_M + row;
        sA[0][row][col] = (col < WMMA_K && gr < Mg && col < K) ?
            LDG(&A[base_A + gr * K + col]) : __float2half(0.0f);
    }
#pragma unroll 4
    for (int i = ty * 32 + tx; i < WMMA_TILE_N * SB_LD; i += 128) {
        int col = i / SB_LD, row = i % SB_LD;
        int gc = tile_n_B + col;
        sB[0][col][row] = (row < WMMA_K && row < K && gc < N) ?
            LDG(&B[base_B + gc * K + row]) : __float2half(0.0f);
    }
    __syncthreads();

    for (int k = 0; k < num_tiles_k; k++) {
        int buf = k & 1;
        int next_buf = 1 - buf;
        int k_next = (k + 1) * WMMA_K;

        if (k + 1 < num_tiles_k) {
#pragma unroll 4
            for (int i = ty * 32 + tx; i < WMMA_TILE_M * SA_LD; i += 128) {
                int row = i / SA_LD, col = i % SA_LD;
                int gr = blockIdx.y * WMMA_TILE_M + row;
                sA[next_buf][row][col] = (col < WMMA_K && gr < Mg && k_next + col < K) ?
                    LDG(&A[base_A + gr * K + k_next + col]) : __float2half(0.0f);
            }
#pragma unroll 4
            for (int i = ty * 32 + tx; i < WMMA_TILE_N * SB_LD; i += 128) {
                int col = i / SB_LD, row = i % SB_LD;
                int gc = tile_n_B + col;
                sB[next_buf][col][row] = (row < WMMA_K && k_next + row < K && gc < N) ?
                    LDG(&B[base_B + gc * K + k_next + row]) : __float2half(0.0f);
            }
        }

        nvcuda::wmma::load_matrix_sync(a_frag, &sA[buf][warpM * WMMA_M][0], SA_LD);
        nvcuda::wmma::load_matrix_sync(b_frag, &sB[buf][warpN * WMMA_N][0], SB_LD);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    __shared__ float sC[WMMA_TILE_M][SC_LD];
    nvcuda::wmma::store_matrix_sync(&sC[warpM * WMMA_M][warpN * WMMA_N], c_frag, SC_LD, nvcuda::wmma::mem_row_major);
    __syncthreads();

#pragma unroll
    for (int i = ty * 32 + tx; i < WMMA_M * WMMA_N; i += 128) {
        int r = i / WMMA_N, c = i % WMMA_N;
        int row = tile_m + r, col = tile_n + c;
        if (row < Mg && col < N)
            C[base_C + row * N + col] = __float2half(sC[warpM * WMMA_M + r][warpN * WMMA_N + c]);
    }
}

// 简化 kernel (仅用于与 CPU 对照验证)
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
               __half2float(B[base_B + col * K + k]);
    }
    C[base_C + row * N + col] = __float2half(sum);
}

int main() {
    srand(42);

    int M_list[NUM_GROUPS];
    int total_M = 0;
    std::cout << "M values: ";
    for (int g = 0; g < NUM_GROUPS; g++) {
        M_list[g] = 251 + (rand() % 10);
        total_M += M_list[g];
        std::cout << M_list[g] << " ";
    }
    std::cout << std::endl;

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

    __half* h_A = (__half*)malloc(size_A);
    __half* h_B = (__half*)malloc(size_B);
    __half* h_C = (__half*)malloc(size_C);
    __half* h_C_ref = (__half*)malloc(size_C);

    for (size_t i = 0; i < total_M * K_DIM; i++)
        h_A[i] = __float2half((float)(rand() % 10) / 100.0f);
    for (int g = 0; g < NUM_GROUPS; g++) {
        int off = offset_B[g];
        for (int n = 0; n < N_DIM; n++)
            for (int k = 0; k < K_DIM; k++)
                h_B[off + n * K_DIM + k] = __float2half((float)(rand() % 10) / 100.0f);
    }

    cpu_group_gemm_ref(h_A, h_B, h_C_ref, M_list, K_DIM, N_DIM);

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

    // ============ CPU 验证正确性 ============
    dim3 block(16, 16);
    dim3 grid;
    for (int g = 0; g < NUM_GROUPS; g++) {
        grid.x = (N_DIM + block.x - 1) / block.x;
        grid.y = (M_list[g] + block.y - 1) / block.y;
        grid.z = 1;
        group_gemm_simple_kernel<<<grid, block>>>(g, d_A, d_B, d_C,
            d_M_list, d_offset_A, d_offset_B, d_offset_C, K_DIM, N_DIM);
    }
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    float max_err_simple = 0.0f;
    for (size_t i = 0; i < total_M * N_DIM; i++)
        max_err_simple = fmaxf(max_err_simple, fabsf(__half2float(h_C_ref[i]) - __half2float(h_C[i])));
    std::cout << "Simple kernel vs CPU max err: " << max_err_simple << std::endl;

    dim3 grid_w((N_DIM + WMMA_TILE_N - 1) / WMMA_TILE_N,
                (260 + WMMA_TILE_M - 1) / WMMA_TILE_M, NUM_GROUPS);
    group_gemm_wmma_fused_kernel<<<grid_w, dim3(WMMA_BLOCK_DIM, WMMA_BLOCK_ROWS)>>>(d_A, d_B, d_C,
        d_M_list, d_offset_A, d_offset_B, d_offset_C, K_DIM, N_DIM);
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    float max_err_wmma = 0.0f;
    int err_count = 0, inf_count = 0;
    for (size_t i = 0; i < total_M * N_DIM; i++) {
        float c_ref = __half2float(h_C_ref[i]);
        float c_gpu = __half2float(h_C[i]);
        if (!isfinite(c_gpu)) inf_count++;
        float err = isfinite(c_ref) && isfinite(c_gpu) ? fabsf(c_ref - c_gpu) : 0.0f;
        if (err > 1e-2f) err_count++;
        max_err_wmma = fmaxf(max_err_wmma, err);
    }

    std::cout << "\n=== 正确性验证 (WMMA vs CPU) ===" << std::endl;
    std::cout << "Max |CPU - GPU|: " << max_err_wmma << ", Error count (tol=1e-2): " << err_count << ", Inf/NaN: " << inf_count << std::endl;
    std::cout << (max_err_wmma < 1e-1f && inf_count == 0 ? "PASS: 结果正确!" : "FAIL") << std::endl;

    // ============ 性能测试 (RTX 4090) ============
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int warmup = 5;
    const int repeat = 10;

    for (int i = 0; i < warmup; i++)
        group_gemm_wmma_fused_kernel<<<grid_w, dim3(WMMA_BLOCK_DIM, WMMA_BLOCK_ROWS)>>>(d_A, d_B, d_C,
            d_M_list, d_offset_A, d_offset_B, d_offset_C, K_DIM, N_DIM);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        group_gemm_wmma_fused_kernel<<<grid_w, dim3(WMMA_BLOCK_DIM, WMMA_BLOCK_ROWS)>>>(d_A, d_B, d_C,
            d_M_list, d_offset_A, d_offset_B, d_offset_C, K_DIM, N_DIM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= repeat;

    double total_flops = 2.0 * (double)total_M * K_DIM * N_DIM;
    double total_bytes = size_A + size_B + size_C;
    double gflops = (total_flops / 1e9) / (ms / 1000.0);
    double gbs = (total_bytes / 1e9) / (ms / 1000.0);

    std::cout << "\n=== 性能 (手写 CUDA, RTX 4090) ===" << std::endl;
    std::cout << "总 M: " << total_M << ", K: " << K_DIM << ", N: " << N_DIM << std::endl;
    std::cout << "单次耗时: " << std::fixed << std::setprecision(4) << ms << " ms" << std::endl;
    std::cout << "计算效率: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS/s" << std::endl;
    std::cout << "内存带宽: " << std::fixed << std::setprecision(2) << gbs << " GB/s" << std::endl;

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

