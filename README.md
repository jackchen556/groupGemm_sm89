# groupGemm_sm89

## 运行结果

```
M values: 257 251 252 252 253 259 252 251 
WMMA (Tensor Core) kernel max err: 0.0078125

=== 正确性验证 ===
Max |CPU - GPU|: 0.0078125
Error count (tol=1e-2): 0, Inf/NaN: 0
PASS: 结果正确!

=== 性能 (总 M: 2027, K: 4096, N: 2048) ===

=== WMMA (Tensor Core + 融合 3D grid) [A100 理论 ~312 TFLOPS] ===
单次耗时: 3.042 ms
计算效率: 11177.87 GFLOPS/s
内存带宽: 52.30 GB/s
```
