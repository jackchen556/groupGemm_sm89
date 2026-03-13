#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

__device__ __forceinline__ __half ld_kernel(const __half* p) {
    return __ldg(p);
}

__device__ __forceinline__ void ld_uint4(const __half* __restrict__ src, __half* __restrict__ dst) {
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
}

#define NUM_GROUPS 8
#define K_DIM 4096
#define N_DIM 2048

#define WMMA_TILE_M 64
#define WMMA_TILE_N 64
#define WMMA_BLOCK_DIM 32
#define WMMA_BLOCK_ROWS 16

#define SA_LD 16
#define SA_BANK_PAD 8
#define SB_LD 16
#define SB_BANK_PAD 8
#define SC_LD 64

// ============ CPU Reference Implementation ============
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
                    sum += __half2float(A[offset_A + m * K + k]) * __half2float(B[offset_B + k + n * N]);
                }
                C[offset_C + m * N + n] = __float2half(sum);
            }
        }
    }
}

// ====== gpu implement =======
// wmma 16*16*16
__global__ __launch_bounds__(512, 2) void group_gemm_wmma_fused_kernel(
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

    __shared__ __half sA[2][WMMA_TILE_M][SA_LD + SA_BANK_PAD];
    __shared__ __half sB[2][WMMA_TILE_N][SB_LD + SB_BANK_PAD];

    int warpM = threadIdx.y / 4;
    int warpN = threadIdx.y % 4;
    int tile_m = blockIdx.y * WMMA_TILE_M;
    int tile_n = blockIdx.x * WMMA_TILE_N;
    int ty = threadIdx.y, tx = threadIdx.x;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int num_tiles_k = K / 16;

    #pragma unroll 1
    for (int t = ty * 32 + tx; t < WMMA_TILE_M * (SA_LD / 8); t += 512) {
        int row = t / (SA_LD / 8), col = (t % (SA_LD / 8)) * 8;
        int gr = tile_m + row;
        if (gr < Mg && col + 7 < K)
            ld_uint4(&A[base_A + gr * K + col], &sA[0][row][col]);
        else {
            for (int i = 0; i < 8; i++)
                sA[0][row][col + i] = (col + i < 16 && gr < Mg && col + i < K) ?
                    ld_kernel(&A[base_A + gr * K + col + i]) : __float2half(0.0f);
        }
    }
    #pragma unroll 1
    for (int t = ty * 32 + tx; t < WMMA_TILE_N * (16 / 8); t += 512) {
        int col = t / (16 / 8), row_base = (t % (16 / 8)) * 8;
        int n_val = tile_n + col;
        if (n_val < N && row_base + 7 < K)
            ld_uint4(&B[base_B + n_val * N + row_base], &sB[0][col][row_base]);
        else {
            for (int i = 0; i < 8; i++)
                sB[0][col][row_base + i] = (n_val < N && row_base + i < K) ?
                    ld_kernel(&B[base_B + n_val * N + row_base + i]) : __float2half(0.0f);
        }
    }
    __syncthreads();

    #pragma unroll 1
    for (int k = 0; k < num_tiles_k; k++) {
        int buf = k & 1;
        int next_buf = 1 - buf;
        int k_next = (k + 1) * 16;

        if (k + 1 < num_tiles_k) {
            #pragma unroll 1
            for (int t = ty * 32 + tx; t < WMMA_TILE_M * (SA_LD / 8); t += 512) {
                int row = t / (SA_LD / 8), col = (t % (SA_LD / 8)) * 8;
                int gr = tile_m + row;
                if (gr < Mg && k_next + col + 7 < K)
                    ld_uint4(&A[base_A + gr * K + k_next + col], &sA[next_buf][row][col]);
                else {
                    for (int i = 0; i < 8; i++)
                        sA[next_buf][row][col + i] = (col + i < 16 && gr < Mg && k_next + col + i < K) ?
                            ld_kernel(&A[base_A + gr * K + k_next + col + i]) : __float2half(0.0f);
                }
            }
            #pragma unroll 1
            for (int t = ty * 32 + tx; t < WMMA_TILE_N * (16 / 8); t += 512) {
                int col = t / (16 / 8), row_base = (t % (16 / 8)) * 8;
                int n_val = tile_n + col;
                if (n_val < N && k_next + row_base + 7 < K)
                    ld_uint4(&B[base_B + n_val * N + k_next + row_base], &sB[next_buf][col][row_base]);
                else {
                    for (int i = 0; i < 8; i++)
                        sB[next_buf][col][row_base + i] = (n_val < N && row_base + i < K) ?
                            ld_kernel(&B[base_B + n_val * N + k_next + row_base + i]) : __float2half(0.0f);
                }
            }
        }

        wmma::load_matrix_sync(a_frag, &sA[buf][warpM * 16][0], SA_LD + SA_BANK_PAD);
        wmma::load_matrix_sync(b_frag, &sB[buf][warpN * 16][0], SB_LD + SB_BANK_PAD);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    __shared__ float sC[WMMA_TILE_M][SC_LD];
    wmma::store_matrix_sync(&sC[warpM * 16][warpN * 16], c_frag, SC_LD, wmma::mem_row_major);
    __syncthreads();

    #pragma unroll 4
    for (int i = ty * 32 + tx; i < WMMA_TILE_M * (WMMA_TILE_N / 8); i += 512) {
        int r = i / (WMMA_TILE_N / 8);
        int c = (i % (WMMA_TILE_N / 8)) * 8;
        int row = tile_m + r, col = tile_n + c;
        if (row < Mg && col + 7 < N) {
            __half h[8];
            #pragma unroll
            for (int j = 0; j < 8; j++)
                h[j] = __float2half(sC[r][c + j]);
            *reinterpret_cast<uint4*>(&C[base_C + row * N + col]) = *reinterpret_cast<uint4*>(h);
        } else {
            for (int j = 0; j < 8 && col + j < N; j++)
                if (row < Mg)
                    C[base_C + row * N + col + j] = __float2half(sC[r][c + j]);
        }
    }
}

int main() {
    srand(26);

    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    std::cout << std::string(60, '=') << std::endl;
    std::cout << "  Group GEMM Benchmark" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "  GPU Device: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    int M_list[NUM_GROUPS];
    int total_M = 0;
    std::cout << "  Problem Config:" << std::endl;
    std::cout << "    M per group: ";
    for (int g = 0; g < NUM_GROUPS; g++) {
        M_list[g] = 251 + (rand() % 10);
        total_M += M_list[g];
        std::cout << M_list[g];
        if (g < NUM_GROUPS - 1) std::cout << ", ";
    }
    std::cout << "  (total M = " << total_M << ")" << std::endl;
    std::cout << "    K = " << K_DIM << ", N = " << N_DIM << std::endl;
    std::cout << std::string(60, '-') << std::endl;

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

    __half* h_B_col = (__half*)malloc(size_B);
    for (int g = 0; g < NUM_GROUPS; g++) {
        int off = offset_B[g];
        for (int n = 0; n < N_DIM; n++)
            for (int k = 0; k < K_DIM; k++)
                h_B_col[off + k + n * N_DIM] = h_B[off + n * K_DIM + k];
    }

    cpu_group_gemm_ref(h_A, h_B_col, h_C_ref, M_list, K_DIM, N_DIM);
    free(h_B);

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
    cudaMemcpy(d_B, h_B_col, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M_list, M_list, NUM_GROUPS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset_A, offset_A, NUM_GROUPS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset_B, offset_B, NUM_GROUPS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset_C, offset_C, NUM_GROUPS * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid_w((N_DIM + WMMA_TILE_N - 1) / WMMA_TILE_N,
                (260 + WMMA_TILE_M - 1) / WMMA_TILE_M, NUM_GROUPS);
    dim3 block_w(WMMA_BLOCK_DIM, WMMA_BLOCK_ROWS);
    group_gemm_wmma_fused_kernel<<<grid_w, block_w>>>(d_A, d_B, d_C,
        d_M_list, d_offset_A, d_offset_B, d_offset_C, K_DIM, N_DIM);
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

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

    std::cout << "  Check CPU vs GPU:" << std::endl;
    std::cout << "    Max |CPU - GPU|:     " << std::scientific << std::setprecision(4) << max_err << std::endl;
    std::cout << "    Error count (1e-2):  " << err_count << std::endl;
    std::cout << "    Inf/NaN count:       " << inf_count << std::endl;
    std::cout << "    Result:              " << (max_err < 1e-1f && inf_count == 0 ? "PASS" : "FAIL") << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int warmup = 20;
    const int repeat = 200;

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
    if (ms < 0.001f) ms = 0.001f;

    double total_flops = 2.0 * (double)total_M * K_DIM * N_DIM;
    double total_bytes = size_A + size_B + size_C;
    double gflops = (total_flops / 1e9) / (ms / 1000.0);
    double gbs = (total_bytes / 1e9) / (ms / 1000.0);

    std::cout << "  Performance (GPU):" << std::endl;
    std::cout << "    Latency:            " << std::fixed << std::setprecision(4) << ms << " ms" << std::endl;
    std::cout << "    Compute throughput:  " << std::setprecision(2) << gflops << " GFLOPS/s" << std::endl;
    std::cout << "    Memory bandwidth:   " << std::setprecision(2) << gbs << " GB/s" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_M_list);
    cudaFree(d_offset_A);
    cudaFree(d_offset_B);
    cudaFree(d_offset_C);
    free(h_A);
    free(h_B_col);
    free(h_C);
    free(h_C_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
