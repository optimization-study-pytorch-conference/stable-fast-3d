# Experiment Order

1) fp32
2) bf16
3) SD3 Fuse QKV Projections
4) bf16 + SDPA
5) bf16 + SDPA + sparsify
6) bf16 + SDPA + torch.compile
7) bf16 + SDPA + torch.compile + sparsify
8) bf16 + SDPA + torch.compile + sparsify + Fuse-QKV
9) int8 + SDPA + torch.compile + sparsify
10) int8 + SDPA + torch.compile + Fuse-QKV  
11) int8 + SDPA + torch.compile + Fuse-QKV + sparsify
12) int4 + SDPA + torch.compile + Fuse-QKV + sparsify
13) bf16 + autoquant-int8 + SDPA + torch.compile  + Fuse-QKV + sparsify
14) bf16 + autoquant-int4 + SDPA + torch.compile  + Fuse-QKV + sparsify
15) bf16 + autoquant-fp8dqrow + SDPA + torch.compile + Fuse-QKV + sparsify
    
Inspiration: 
https://github.com/sayakpaul/diffusers-torchao/blob/main/inference/benchmark_video.py