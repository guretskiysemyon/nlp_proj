#!/bin/bash
#SBATCH --partition=studentkillable
#SBATCH --gpus=1
#SBATCH --output=gpu_check_%j.out
#SBATCH --account=gpu-students

cd /home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
