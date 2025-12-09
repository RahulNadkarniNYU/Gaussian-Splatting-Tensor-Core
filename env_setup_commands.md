srun --nodes=1 --gres=gpu:a100:1 --time=2:00:00 --mem=32G --pty /bin/bash
or
srun  --nodes=1 \
      --gres=gpu:rtx8000:1 \
      --time=6:00:00 \
      --mem=24G \
      --pty /bin/bash
(rtx preferred!)

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash

source /ext3/env.sh

In exp_env:
conda activate exp_env
cd /scratch/sr7463/BDML_project/Gaussian-Splatting-Tensor-Core

rm -rf submodules/tcgs_speedy_rasterizer/build
rm -rf submodules/tcgs_speedy_rasterizer/*.egg-info
rm -rf submodules/tcgs_speedy_rasterizer/__pycache__

pip cache purge
pip uninstall -y diff_gaussian_rasterization
pip install --no-build-isolation --force-reinstall ./submodules/tcgs_speedy_rasterizer

#############################################


conda activate tcgs_env


#############################################

One time Install commands:
conda activate 
conda create -n tcgs_env python=3.9 -y
conda activate tcgs_env
pip install --upgrade pip setuptools wheel ninja
pip install torch==1.13.1+cu117 \
            torchvision==0.14.1+cu117 \
            torchaudio==0.13.1 \
            --extra-index-url https://download.pytorch.org/whl/cu117
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pip install plyfile tqdm opencv-python joblib
pip install --no-build-isolation ./submodules/tcgs_speedy_rasterizer
pip install --no-build-isolation ./submodules/simple-knn
pip install --no-build-isolation ./submodules/fused-ssim
pip install --upgrade "numpy<2"
