#!/bin/bash
# gpu数量
num_gpus=8
# 只显示echo内容
echo_only=0

if [[ echo_only -eq 1 ]]; then
    # 使⽤sinfo获取每个partition及其节点状态的列表
    sinfo --format="%R %T %N" 2>/dev/null | tail -n +2 | while read -r partition state nodes; do
        # 跳过状态为drain, down, 或 inval的partitions
        if [[ $state == "drain" || $state == "down" || $state == "inval" ]]; then
            continue
        fi
        echo ">>> Partition: $partition - State: $state <<<"
        # 使⽤scontrol show hostname转换为单独的节点名
        scontrol show hostname "$nodes" 2>/dev/null | while read -r node; do
            # 如果节点名中包含"gpu"，则尝试执行nvidia-smi
            if [[ $node == *gpu* || $node == *bigmem* ]]; then
                # 使用timeout限制srun命令的最大执行时间
                if timeout 2s srun -p $partition -w $node --gres=gpu:$num_gpus --pty nvidia-smi > /dev/null 2>&1; then
                    echo ""
                    echo "************* node available *************"
                    echo "#SBATCH --partition=$partition"
                    echo "#SBATCH --nodelist=$node"
                    echo "******************************************"
                    echo ""
                fi
            fi
        done
    done
else
    # 使⽤sinfo获取每个partition及其节点状态的列表
    sinfo --format="%R %T %N" | tail -n +2 | while read -r partition state nodes; do
        # 跳过状态为drain, down, 或 inval的partitions
        if [[ $state == "drain" || $state == "down" || $state == "inval" ]]; then
            continue
        fi
        echo ">>> Partition: $partition - State: $state <<<"
        # 使⽤scontrol show hostname转换为单独的节点名
        scontrol show hostname "$nodes" | while read -r node; do
            # 如果节点名中包含"gpu"，则尝试执行nvidia-smi
            if [[ $node == *gpu* || $node == *bigmem* ]]; then
                # 使用timeout限制srun命令的最大执行时间
                echo "=== Node: $node ==="
                if timeout 2s srun -p $partition -w $node --gres=gpu:$num_gpus --pty nvidia-smi; then
                    echo ""
                    echo "************* node available *************"
                    echo "#SBATCH --partition=$partition"
                    echo "#SBATCH --nodelist=$node"
                    echo "******************************************"
                    echo ""
                fi
            fi
        done
    done
fi