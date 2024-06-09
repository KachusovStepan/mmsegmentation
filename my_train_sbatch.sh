#!/bin/bash
export CUDA_HOME='/opt/cuda-11.5'
module load gpu/cuda-11.5
conda activate openmmlab

SBATCH_CPU="-n1 -p v100 --gres=gpu:v100:2"
SBATCH_CPU="-p debug"

sbatch \
--cpus-per-task=1 \
--mem=32000 \
$SBATCH_CPU \
-t 00:30:00 \
--job-name=mmsegmentation \
--output=./logs/"mmsegmentation_Dataset_landcover.ai_512_128_CrossEntropyLoss_SGD_bsize_2_%j" \
\
--wrap="python3.8 -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 ./train.py ./configs/config.py --launcher pytorch"
