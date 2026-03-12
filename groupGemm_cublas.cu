/**
 * Group GEMM - cuBLAS 版本
 * C[g] = A[g] * B[g]^T，与 group_gemm_u4.cu 逻辑一致
 * 编译: nvcc -O3 -arch=sm_89 --use_fast_math -o group_gemm_cublas group_gemm_cublas.cu -lcublas
 */
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define NG 8
#define K_DIM 4096
#define N_DIM 2048

void cpu_ref(const __half* A, const __half* B, __half* C, const int* M, int K, int N) {
    for (int g = 0; g < NG; g++) {
        int oA = 0, oB = 0, oC = 0;
        for (int i = 0; i < g; i++) { oA += M[i]*K; oB += K*N; oC += M[i]*N; }
        for (int m = 0; m < M[g]; m++)
            for (int n = 0; n < N; n++) {
                float s = 0;
                for (int k = 0; k < K; k++) s += __half2float(A[oA+m*K+k]) * __half2float(B[oB+k+n*N]);
                C[oC+m*N+n] = __float2half(s);
            }
    }
}

int main() {
    srand(42);
    int M[NG], total_M = 0, oA[NG], oB[NG], oC[NG];
    for (int g = 0; g < NG; g++) { M[g] = 251 + (rand()%10); total_M += M[g]; }
    oA[0]=oB[0]=oC[0]=0;
    for (int g = 1; g < NG; g++) { oA[g]=oA[g-1]+M[g-1]*K_DIM; oB[g]=oB[g-1]+K_DIM*N_DIM; oC[g]=oC[g-1]+M[g-1]*N_DIM; }

    std::cout << "M: "; for (int g=0; g<NG; g++) std::cout << M[g] << " "; std::cout << std::endl;

    size_t szA = (size_t)total_M*K_DIM*sizeof(__half), szB = (size_t)NG*K_DIM*N_DIM*sizeof(__half), szC = (size_t)total_M*N_DIM*sizeof(__half);
    __half *h_A = (__half*)malloc(szA), *h_B = (__half*)malloc(szB), *h_C_ref = (__half*)malloc(szC);

    for (size_t i = 0; i < total_M*K_DIM; i++) h_A[i] = __float2half((float)(rand()%10)/100.f);
    for (int g = 0; g < NG; g++)
        for (int n = 0; n < N_DIM; n++) for (int k = 0; k < K_DIM; k++)
            h_B[oB[g]+k+n*N_DIM] = __float2half((float)(rand()%10)/100.f);

    cpu_ref(h_A, h_B, h_C_ref, M, K_DIM, N_DIM);

    // FP32 验证 (cublasSgemm 保证正确)
    size_t szAf = total_M*K_DIM*sizeof(float), szBf = NG*K_DIM*N_DIM*sizeof(float), szCf = total_M*N_DIM*sizeof(float);
    float *h_Af = (float*)malloc(szAf), *h_BTf = (float*)malloc(szBf);
    for (size_t i = 0; i < total_M*K_DIM; i++) h_Af[i] = __half2float(h_A[i]);
    for (int g = 0; g < NG; g++)
        for (int n = 0; n < N_DIM; n++) for (int k = 0; k < K_DIM; k++)
            h_BTf[oB[g]+k+n*K_DIM] = __half2float(h_B[oB[g]+k+n*N_DIM]);

    float *h_Ac = (float*)malloc(szAf);
    for (int g = 0; g < NG; g++)
        for (int k = 0; k < K_DIM; k++) for (int m = 0; m < M[g]; m++)
            h_Ac[oA[g]+m+k*M[g]] = h_Af[oA[g]+m*K_DIM+k];

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, szAf); cudaMalloc(&d_B, szBf); cudaMalloc(&d_C, szCf);
    cudaMemcpy(d_A, h_Ac, szAf, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_BTf, szBf, cudaMemcpyHostToDevice);

    cublasHandle_t h; cublasCreate(&h);
    float alpha = 1.f, beta = 0.f;

    for (int g = 0; g < NG; g++)
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M[g], N_DIM, K_DIM, &alpha, d_A+oA[g], M[g], d_B+oB[g], K_DIM, &beta, d_C+oC[g], M[g]);

    float *h_Cf = (float*)malloc(szCf);
    cudaMemcpy(h_Cf, d_C, szCf, cudaMemcpyDeviceToHost);

    __half *h_Crow = (__half*)malloc(szC);
    for (int g = 0; g < NG; g++)
        for (int n = 0; n < N_DIM; n++) for (int m = 0; m < M[g]; m++)
            h_Crow[oC[g]+m*N_DIM+n] = __float2half(h_Cf[oC[g]+m+n*M[g]]);

    float maxE = 0; int errCnt = 0;
    for (size_t i = 0; i < total_M*N_DIM; i++) {
        float e = fabsf(__half2float(h_C_ref[i]) - __half2float(h_Crow[i]));
        if (e > 1e-2f) errCnt++;
        maxE = fmaxf(maxE, e);
    }
    std::cout << "\n验证: Max|diff|=" << maxE << ", errs=" << errCnt << " " << (maxE<0.1f?"PASS":"FAIL") << std::endl;

    // FP16 性能
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    __half *h_Ac_h = (__half*)malloc(szA), *h_BT_h = (__half*)malloc(szB);
    for (int g = 0; g < NG; g++) {
        for (int k = 0; k < K_DIM; k++) for (int m = 0; m < M[g]; m++) h_Ac_h[oA[g]+m+k*M[g]] = h_A[oA[g]+m*K_DIM+k];
        for (int n = 0; n < N_DIM; n++) for (int k = 0; k < K_DIM; k++) h_BT_h[oB[g]+k+n*K_DIM] = h_B[oB[g]+k+n*N_DIM];
    }
    __half *d_A_h, *d_B_h, *d_C_h;
    cudaMalloc(&d_A_h, szA); cudaMalloc(&d_B_h, szB); cudaMalloc(&d_C_h, szC);
    cudaMemcpy(d_A_h, h_Ac_h, szA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_h, h_BT_h, szB, cudaMemcpyHostToDevice);

    __half alpha_h = __float2half(1.f), beta_h = __float2half(0.f);
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    for (int i = 0; i < 20; i++)
        for (int g = 0; g < NG; g++)
            cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M[g], N_DIM, K_DIM, &alpha_h, d_A_h+oA[g], CUDA_R_16F, M[g], d_B_h+oB[g], CUDA_R_16F, K_DIM, &beta_h, d_C_h+oC[g], CUDA_R_16F, M[g], CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int i = 0; i < 200; i++)
        for (int g = 0; g < NG; g++)
            cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M[g], N_DIM, K_DIM, &alpha_h, d_A_h+oA[g], CUDA_R_16F, M[g], d_B_h+oB[g], CUDA_R_16F, K_DIM, &beta_h, d_C_h+oC[g], CUDA_R_16F, M[g], CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= 200;

    std::cout << "耗时: " << std::fixed << std::setprecision(4) << ms << " ms, "
              << (2.0*total_M*K_DIM*N_DIM/1e12)/(ms/1000) << " TFLOPS, " << ((szA+szB+szC)/1e9)/(ms/1000) << " GB/s" << std::endl;

    cublasDestroy(h);
    cudaFree(d_A_h); cudaFree(d_B_h); cudaFree(d_C_h);
    free(h_A); free(h_B); free(h_C_ref); free(h_Af); free(h_BTf); free(h_Ac); free(h_Cf); free(h_Crow); free(h_Ac_h); free(h_BT_h);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
