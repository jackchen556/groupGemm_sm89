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
Simple kernel vs CPU max err: 0

=== 正确性验证 (WMMA + uint4 C 写回 vs CPU) ===
Max |CPU - GPU|: 0.0078125, Error count (tol=1e-2): 0, Inf/NaN: 0
WMMA vs Simple max err: 0.0078125
PASS: 结果正确!

=== 性能 (C 矩阵 uint4 向量化, RTX 4090) ===
总 M: 2027, K: 4096, N: 2048
单次耗时: 1.2708 ms
计算效率: 26760.77 GFLOPS/s
内存带宽: 125.22 GB/s
```
