# groupGemm_sm89

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

=== 正确性验证 (CPU vs WMMA GPU) ===
Max |CPU - GPU|: 0.0078125, Error count (tol=1e-2): 0, Inf/NaN: 0
PASS: 结果正确!

=== 性能 (C 矩阵 uint4 向量化, RTX 4090) ===
总 M: 2027, K: 4096, N: 2048
单次耗时: 0.9814 ms
计算效率: 34651.20 GFLOPS/s
内存带宽: 162.14 GB/s
```
