#!/usr/bin/bash

#SBATCH -J llm4stg_test # 指定任务名称，⽅便追踪
#SBATCH --mem=48G # 指定任务运⾏需要的RAM
#SBATCH --partition=i64m1tga800u
#SBATCH --nodelist=gpu1-1
#SBATCH --gres=gpu:8 # 指定需要的GPU数量
#SBATCH --output=./output/job_%j_%x.out # 指定规范输出的格式
#SBATCH --error=./output/job_%j_%x.err # 指定错误输出的格式
#SBATCH --time=7-00:00:00 # 指定任务运⾏的上限时间（我们的HPC为7天）
#SBATCH --nodes=1 # 指定需要的节点数量（⼀般为1）
#SBATCH --ntasks-per-node=1 # 指定⼀个节点运⾏的任务（⼀般为1）
#SBATCH --cpus-per-task=64 # 指定任务需要的CPU核数
#- Log information （⾮必需）
echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"
#- Load environments 加载Anaconda环境
echo "Loading conda:"
module load anaconda3 # 激活anaconda
source /hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh # 让bash激活 
conda
conda activate llm4stg # 激活环境
echo "Conda ready, currently in env: $CONDA_DEFAULT_ENV"
echo "GPU available: ${CUDA_VISIBLE_DEVICES}"
cd ../stllm # 更改⼯作⽬录
echo "==========+++++ model output below +++++=========="
torchrun --nproc_per_node=8 train.py --expid 'exp4.3_12layers' --temp_prefix --gpt_layers 12 --batch_size 2