# groupGemm_sm89

Test Configuration
Parameter	Value
M Values	257, 251, 252, 252, 253, 259, 252, 251
Total M	2027
K	4096
N	2048
GPU	NVIDIA A100
Theoretical Peak	~312 TFLOPS
Correctness Verification
Metric	Value
WMMA Kernel Max Error	0.0078125
Max |CPU - GPU|	0.0078125
Error Count (tol=1e-2)	0
Inf/NaN Count	0
Status	✅ PASS
Performance Results
WMMA (Tensor Core + Fused 3D Grid)
Metric	Value
Single Run Latency	3.042 ms
Compute Efficiency	11177.87 GFLOPS/s
Memory Bandwidth	52.30 GB/s
Summary
✅ Correctness: All tests passed with error tolerance 1e-2
🚀 Performance: Achieved 11.18 TFLOPS on A100 GPU
