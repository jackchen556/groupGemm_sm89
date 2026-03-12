# groupGemm_sm89

## 运行结果

```
M values: 257 251 252 252 253 259 252 251 
Simple kernel vs CPU max err: 0

=== 正确性验证 (WMMA vs CPU) ===
Max |CPU - GPU|: 0.0078125, Error count (tol=1e-2): 0, Inf/NaN: 0
WMMA vs Simple max err: 0.0078125 (应为 0，否则 B 布局可能不一致)
PASS: 结果正确!

=== 性能 (手写 CUDA, RTX 4090) ===
总 M: 2027, K: 4096, N: 2048
单次耗时: 1.3470 ms
计算效率: 25246.54 GFLOPS/s
内存带宽: 118.13 GB/s
```
