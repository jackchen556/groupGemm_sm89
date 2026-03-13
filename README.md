# groupGemm_sm89


## 运行结果

```

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
