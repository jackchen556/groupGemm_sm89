/**
 * Group GEMM - cuBLAS 版本
 * 与 group_gemm_u4.cu 逻辑一致：8 对矩阵乘法 C[g] = A[g] * B[g]
 * - 输入输出格式相同
 * - CPU 参考实现 + GPU (cuBLAS) 精度验证
 * - 性能指标用于与手写 WMMA 版本对比
 *
 * 编译 (需链接 cuBLAS):
 *   nvcc -O3 -arch=sm_89 --use_fast_math -o group_gemm_cublas group_gemm_cublas.cu -lcublas
 *   A100: nvcc -O3 -arch=sm_80 --use_fast_math -o group_gemm_cublas group_gemm_cublas.cu -lcublas
 * 若链接失败，可加库路径: -L/usr/local/cuda/lib64 -lcublas
 */

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define NUM_GROUPS 8
#define K_DIM 4096
#define N_DIM 2048

// ============ CPU 参考实现 (与 group_gemm_u4.cu 完全相同) ============
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

// A 转置: row-major -> column-major (cuBLAS 默认列主序)
void transpose_A_to_colmajor(const __half* A_row, __half* A_col,
    const int* M_list, const int* offset_A, int K)
{
    for (int g = 0; g < NUM_GROUPS; g++) {
        int Mg = M_list[g];
        int off = offset_A[g];
        for (int k = 0; k < K; k++) {
            for (int m = 0; m < Mg; m++) {
                A_col[off + m + k * Mg] = A_row[off + m * K + k];
            }
        }
    }
}

// B_col 转为 cuBLAS 格式: C = A * B^T，需传 B (N×K) 列主序，用 op(B)=T
// B_col[k+n*N]=B[n,k]，cuBLAS 期望 B[n,k] 在 n+k*N (N×K 列主序)
void convert_B_for_cublas(const __half* B_col, __half* B_cublas,
    const int* offset_B, int K, int N)
{
    for (int g = 0; g < NUM_GROUPS; g++) {
        int off = offset_B[g];
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                B_cublas[off + n + k * N] = B_col[off + k + n * N];
            }
        }
    }
}

// C 转置: column-major -> row-major (cuBLAS 输出列主序，需转回行主序以便与 CPU 对比)
void transpose_C_to_rowmajor(const __half* C_col, __half* C_row,
    const int* M_list, const int* offset_C, int N)
{
    for (int g = 0; g < NUM_GROUPS; g++) {
        int Mg = M_list[g];
        int off = offset_C[g];
        for (int n = 0; n < N; n++) {
            for (int m = 0; m < Mg; m++) {
                C_row[off + m * N + n] = C_col[off + m + n * Mg];
            }
        }
    }
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
    __half* h_C_ref = (__half*)malloc(size_C);
    __half* h_C_cublas = (__half*)malloc(size_C);

    // 初始化 A (row-major)
    for (size_t i = 0; i < total_M * K_DIM; i++)
        h_A[i] = __float2half((float)(rand() % 10) / 100.0f);

    // 初始化 B 原始布局 B[n][k]，再转置为 B_col[k+n*N]=B[n][k]
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
    free(h_B);

    // CPU 参考
    cpu_group_gemm_ref(h_A, h_B_col, h_C_ref, M_list, K_DIM, N_DIM);

    // A 转置为列主序供 cuBLAS 使用
    __half* h_A_col = (__half*)malloc(size_A);
    transpose_A_to_colmajor(h_A, h_A_col, M_list, offset_A, K_DIM);

    // B 转为 cuBLAS 格式: C = A * B^T，传 B (N×K) 列主序，用 op(B)=T
    __half* h_B_cublas = (__half*)malloc(size_B);
    convert_B_for_cublas(h_B_col, h_B_cublas, offset_B, K_DIM, N_DIM);

    // GPU 分配
    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A_col, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_cublas, size_B, cudaMemcpyHostToDevice);

    // cuBLAS 初始化
    cublasHandle_t handle;
    cublasCreate(&handle);

    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    // ============ cuBLAS 执行 8 组 GEMM ============
    // C = A * B^T，A 为 M×K 列主序，B 为 N×K 列主序，用 op(B)=T
    for (int g = 0; g < NUM_GROUPS; g++) {
        int M = M_list[g];
        int K = K_DIM;
        int N = N_DIM;
        const __half* pA = d_A + offset_A[g];
        const __half* pB = d_B + offset_B[g];
        __half* pC = d_C + offset_C[g];

        cublasStatus_t st = cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            M, N, K,
            &alpha,
            pA, CUDA_R_16F, M,
            pB, CUDA_R_16F, N,
            &beta,
            pC, CUDA_R_16F, M,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasGemmEx failed: " << st << std::endl;
            return 1;
        }
    }

    // cuBLAS 输出为列主序，需转置为行主序
    __half* h_C_col = (__half*)malloc(size_C);
    cudaMemcpy(h_C_col, d_C, size_C, cudaMemcpyDeviceToHost);
    transpose_C_to_rowmajor(h_C_col, h_C_cublas, M_list, offset_C, N_DIM);
    free(h_C_col);

    // ============ 正确性验证 (CPU vs cuBLAS GPU) ============
    float max_err = 0.0f;
    int err_count = 0, inf_count = 0;
    for (size_t i = 0; i < total_M * N_DIM; i++) {
        float c_ref = __half2float(h_C_ref[i]);
        float c_gpu = __half2float(h_C_cublas[i]);
        if (!isfinite(c_gpu)) inf_count++;
        float err = isfinite(c_ref) && isfinite(c_gpu) ? fabsf(c_ref - c_gpu) : 0.0f;
        if (err > 1e-2f) err_count++;
        max_err = fmaxf(max_err, err);
    }

    std::cout << "\n=== 正确性验证 (CPU vs cuBLAS GPU) ===" << std::endl;
    std::cout << "Max |CPU - GPU|: " << max_err << ", Error count (tol=1e-2): " << err_count << ", Inf/NaN: " << inf_count << std::endl;
    std::cout << (max_err < 1e-1f && inf_count == 0 ? "PASS: 结果正确!" : "FAIL") << std::endl;

    // ============ 性能测试 ============
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int warmup = 20;
    const int repeat = 200;

    for (int i = 0; i < warmup; i++) {
        for (int g = 0; g < NUM_GROUPS; g++) {
            int M = M_list[g];
            const __half* pA = d_A + offset_A[g];
            const __half* pB = d_B + offset_B[g];
            __half* pC = d_C + offset_C[g];
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N_DIM, K_DIM,
                &alpha, pA, CUDA_R_16F, M, pB, CUDA_R_16F, N_DIM,
                &beta, pC, CUDA_R_16F, M,
                CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        for (int g = 0; g < NUM_GROUPS; g++) {
            int M = M_list[g];
            const __half* pA = d_A + offset_A[g];
            const __half* pB = d_B + offset_B[g];
            __half* pC = d_C + offset_C[g];
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N_DIM, K_DIM,
                &alpha, pA, CUDA_R_16F, M, pB, CUDA_R_16F, N_DIM,
                &beta, pC, CUDA_R_16F, M,
                CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }
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

    std::cout << "\n=== 性能 (cuBLAS FP16 Tensor Core) ===" << std::endl;
    std::cout << "总 M: " << total_M << ", K: " << K_DIM << ", N: " << N_DIM << std::endl;
    std::cout << "单次耗时: " << std::fixed << std::setprecision(4) << ms << " ms" << std::endl;
    std::cout << "计算效率: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS/s";
    std::cout << " (" << std::fixed << std::setprecision(2) << (gflops / 1000.0) << " TFLOPS)" << std::endl;
    std::cout << "内存带宽: " << std::fixed << std::setprecision(2) << gbs << " GB/s" << std::endl;

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_A_col);
    free(h_B_col);
    free(h_B_cublas);
    free(h_C_ref);
    free(h_C_cublas);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
