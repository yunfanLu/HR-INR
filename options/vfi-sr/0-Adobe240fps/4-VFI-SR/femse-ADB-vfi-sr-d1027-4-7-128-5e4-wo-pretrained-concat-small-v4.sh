#!/bin/bash

#SBATCH -N 1 # 指定node的数量
#SBATCH -p i64m1tga800u #  i64m1tga40u, i64m1tga800u
#SBATCH --gres=gpu:1 # 需要使用多少GPU，n是需要的数量
#SBATCH --time=7-00:00:00 # 指定任务运⾏的上限时间（我们的HPC为7天
#SBATCH --job-name=rstt-TL-vfi-d1020-4-7-256-pretrained-large-v4
#SBATCH --output=./hpc-ii/%j-job.out # slurm的输出文件，%j是jobid
#SBATCH --error=./hpc-ii/%j-error.err # 指定错误输出的格式
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


python egrsdb/main.py \
  --yaml_file="options/vfi-sr/0-Adobe240fps/4-VFI-SR/femse-ADB-vfi-sr-d1027-4-7-128-5e4-wo-pretrained-concat-small-v4.yaml" \
  --log_dir="./log/vfi-sr/0-Adobe240fps/4-VFI-SR/femse-ADB-vfi-sr-d1027-4-7-128-5e4-wo-pretrained-concat-small-v4/" \
  --alsologtostderr=True

