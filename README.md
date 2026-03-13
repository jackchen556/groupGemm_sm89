# groupGemm_sm89


## 运行结果

```
============================================================
  Group GEMM Benchmark
============================================================
  GPU Device: NVIDIA GeForce RTX 4090
  Compute Capability: 8.9
  Global Memory: 24080 MB
------------------------------------------------------------
  Problem Config:
    M per group: 258, 251, 253, 252, 254, 254, 253, 260  (total M = 2035)
    K = 4096, N = 2048
------------------------------------------------------------
  Check CPU vs GPU:
    Max |CPU - GPU|:     7.8125e-03
    Error count (1e-2):  0
    Inf/NaN count:       0
    Result:              PASS
------------------------------------------------------------
  Performance (GPU):
    Latency:            0.6431 ms
    Compute throughput:  53085.20 GFLOPS/s
    Memory bandwidth:   247.57 GB/s
============================================================
```
