# groupGemm_sm89

RTX 4090 硬件规格：
| 指标              | 数值        |
|-------------------|-------------|
| FP16 Tensor 算力  | 80 TFLOPS   |
| 显存大小          | 24GB GDDR6X |


## 运行结果

```
========================
cublas:
M: 257 251 252 252 253 259 252 251 

验证: Max|diff|=0.0078125, errs=0 PASS
耗时: 0.4579 ms, 74.2721 TFLOPS, 347.5295 GB/s

========================

cuda:

M values: 257 251 252 252 253 259 252 251 

=== Correctness Check (CPU vs WMMA GPU, 64_64 tile) ===
Max |CPU - GPU|: 0.0078125, Error count (tol=1e-2): 0, Inf/NaN: 0
PASS: Results match!

=== Performance (64x64 tile, C uint4 vectorized, RTX 4090) ===
Total M: 2027, K: 4096, N: 2048
Latency per kernel: 0.6621 ms
Compute throughput: 51360.74 GFLOPS/s
Memory bandwidth: 240.32 GB/s
```
