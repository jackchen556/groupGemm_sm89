# groupGemm_sm89

## 运行结果

```
M values: 257 251 252 252 253 259 252 251 
Simple kernel vs CPU max err: 0

=== 正确性验证 (WMMA vs CPU) ===
Max |CPU - GPU|: 0.0078125, Error count (tol=1e-2): 0, Inf/NaN: 0
PASS: 结果正确!

=== 性能 (手写 CUDA, RTX 4090) ===
总 M: 2027, K: 4096, N: 2048
单次耗时: 1.3677 ms
计算效率: 24864.93 GFLOPS/s
内存带宽: 116.35 GB/s
```
