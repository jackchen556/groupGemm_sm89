# groupGemm_sm89


## 运行结果

```

============================================================
  Group GEMM Benchmark
============================================================
  GPU Device: NVIDIA GeForce RTX 4090 D
  Compute Capability: 8.9
  Global Memory: 24080 MB
------------------------------------------------------------
  Problem Config:
    M per group: 257, 251, 252, 252, 253, 259, 252, 251  (total M = 2027)
    K = 4096, N = 2048
------------------------------------------------------------
  Check CPU vs GPU:
    Max |CPU - GPU|:     7.8125e-03
    Error count (1e-2):  0
    Inf/NaN count:       0
    Result:              PASS
------------------------------------------------------------
  Performance (GPU):
    Latency:            0.7680 ms
    Compute throughput:  44283.15 GFLOPS/s
    Memory bandwidth:   207.21 GB/s
============================================================
```
