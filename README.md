## Optimizations to TCGS
TC-GS: https://arxiv.org/pdf/2505.24796v2

This repository contains our optimized implementation of Tensor Core Gaussian Splatting (TC-GS) for inference-time rendering, along with optional inference-only culling strategies for further FPS improvements.
The work focuses exclusively on rendering (not training) and is based on systematic profiling and optimization of the TC-GS pipeline.

What This Code Does (Read the report pdf for more details)

1. System-Level TC-GS Optimizations (Baseline)
We implement and evaluate the following core system optimizations:
- CUDA Graphs to eliminate kernel launch overhead
- Static buffer reuse to remove per-frame cudaMalloc/cudaFree
- Restructured WMMA kernels with improved instruction-level parallelism
- Warp coherency optimizations for better memory access and reduced divergence
These changes improve FPS while preserving identical rendering quality (PSNR) compared to baseline TC-GS 

2. Inference-Time Gaussian Culling (Optional)
On top of the optimized TC-GS baseline, we provide lightweight, inference-only culling methods that reduce the number of Gaussians processed per frame:
- Opacity culling (safe, quality-preserving so same PSNR)
- Screen-space radius culling (removes sub-pixel Gaussians)
- View frustum culling
- Distance-based LOD culling
These methods trade off FPS vs PSNR and are intended for controlled benchmarking and analysis.

3. Profiling: But that's just one nsys and nvprof command.

## Setup
1. Download MipNERF 360 dataset.

2. 3DGS paper: https://arxiv.org/abs/2308.04079
We first train the model on bonsai scene from MipNERF360 using the 3DGS repo (Ours and TCGS is an inference rendering only method). Our trained bonsai scene model is available here: https://drive.google.com/drive/folders/1APRkVU5F-UoMBPMscbnDBfDyrekWVhbu?usp=sharing

3. Clone 3DGS and TC-GS and clone ours: git clone git@github.com:RahulNadkarniNYU/Gaussian-Splatting-Tensor-Core.git --recursive
Environment setup for TC-GS and ours can be found in env_setup_commands.md

4. Benchmark inference of 3DGS, TC-GS render.py
Then just change paths and change permissions and run ./eval.sh in our repo.


## Key Results (MipNeRF-360 Bonsai)
- 330.7 FPS with system-level TC-GS optimizations
- 336.2 FPS with opacity culling (same PSNR)
- Up to 340+ FPS with additional LOD/radius culling (minor PSNR drop)
- 342.1% speedup over original CUDA-core 3DGS and 6.6% speedup over TC-GS with no PSNR quality loss.
