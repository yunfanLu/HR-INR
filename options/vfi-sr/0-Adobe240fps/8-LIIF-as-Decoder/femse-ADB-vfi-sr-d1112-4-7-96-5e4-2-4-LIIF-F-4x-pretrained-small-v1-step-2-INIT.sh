#!/bin/bash

#SBATCH -N 1 # 指定node的数量
#SBATCH -p i64m1tga40u #  i64m1tga40u, i64m1tga800u, aiperf
#SBATCH --gres=gpu:1 # 需要使用多少GPU，n是需要的数量
#SBATCH --time=7-00:00:00 # 指定任务运⾏的上限时间（我们的HPC为7天
#SBATCH --job-name=rstt-TL-vfi-d1020-4-7-256-pretrained-large-v4
#SBATCH --output=./hpc-ii-LIIF/%j-job.out # slurm的输出文件，%j是jobid
#SBATCH --error=./hpc-ii-LIIF/%j-error.err # 指定错误输出的格式
#SBATCH --cpus-per-task=8


export CUDA_VISIBLE_DEVICES="0"
export PATH="/hpc2hdd/home/ylu066/miniconda3/bin/":$PATH
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"
echo "Python version: $(python --version)"
echo "              : $(which python)"
echo "CUDA version: $(nvcc --version)"
nvidia-smi

# options/vfi-sr/0-Adobe240fps/8-LIIF-as-Decoder/femse-ADB-vfi-sr-d1112-4-7-96-5e4-2-4-LIIF-F-4x-pretrained-small-v1-step-2-INIT.sh

# The pretrain model is in vfi-sr/0-Adobe240fps/4-VFI-SR/femse-ADB-vfi-sr-d1030-4-7-128-5e4-2-4-wo-pretrained-small-v1-resume

python egrsdb/main.py \
  --yaml_file="options/vfi-sr/0-Adobe240fps/8-LIIF-as-Decoder/femse-ADB-vfi-sr-d1112-4-7-96-5e4-2-4-LIIF-F-4x-pretrained-small-v1.yaml" \
  --log_dir="./log/vfi-sr/0-Adobe240fps/8-LIIF-as-Decoder/femse-ADB-vfi-sr-d1112-4-7-96-5e4-2-4-LIIF-F-4x-pretrained-small-v1-step-2-INIT/" \
  --alsologtostderr=True
  #  \
  # --RESUME_PATH="./log/vfi-sr/0-Adobe240fps/4-VFI-SR/femse-ADB-vfi-sr-d1030-4-7-128-5e4-2-4-wo-pretrained-small-v1/checkpoint.pth.tar" \
  # --RESUME_SET_EPOCH=True

