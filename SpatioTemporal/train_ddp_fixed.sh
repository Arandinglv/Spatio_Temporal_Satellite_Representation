#!/bin/bash

# DDP训练脚本 - SOTA级多卡时空对比学习
# 使用方法:
#   bash train_ddp_fixed.sh                    # 使用默认GPU 0,1,2,3
#   bash train_ddp_fixed.sh "0,1,2,3"         # 指定特定GPU
#   bash train_ddp_fixed.sh "4,5,6,7"         # 使用其他GPU

# GPU配置 (可通过第一个参数指定)
GPUS=${1:-"0,1"}  # 默认使用GPU 0,1
export CUDA_VISIBLE_DEVICES=${GPUS}
NUM_GPUS=$(echo ${GPUS} | tr ',' '\n' | wc -l)

echo "======================================"
echo "📊 DDP Training Configuration (SOTA)"
echo "======================================"
echo "🔥 GPUs: ${GPUS} (${NUM_GPUS} cards)"
echo "💾 Data: /hpc2hdd/home/qwang650/project/yutianjiang/stmodel/dataset/Guangzhou"
echo "📈 Total effective batch size: $((20 * NUM_GPUS * 4 * 2))"
echo "🎯 Per GPU batch size: 20"
echo "🧠 Contrastive Learning enhanced by ${NUM_GPUS}x negative samples"
echo "======================================"

# 验证GPU可用性
echo "🔍 Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found"
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits | head -n ${NUM_GPUS}

# 验证数据路径
DATA_PATH="/hpc2hdd/home/qwang650/project/yutianjiang/stmodel/dataset/Guangzhou"
if [ ! -d "${DATA_PATH}" ]; then
    echo "❌ ERROR: Data path not found: ${DATA_PATH}"
    exit 1
fi

echo "✅ All checks passed. Starting training..."

# 创建实验名称
EXPERIMENT_NAME="guangzhou_spatiotemporal_ddp_$(date +%Y%m%d_%H%M)"

# 使用torchrun启动DDP训练 (完全按照原脚本参数)
torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    run.py \
    --data /hpc2hdd/home/qwang650/project/yutianjiang/stmodel/dataset/Guangzhou \
    --dataset-name STlabelScoreGroupDataset \
    --arch resnet50 \
    --epochs 100 \
    --batch-size 20 \
    --lr 0.0003 \
    --weight-decay 1e-4 \
    --seed 42 \
    --num_neg 6 \
    --workers 12 \
    --log_root /hpc2hdd/home/qwang650/project/yutianjiang/stmodel/training_results \
    --wandb-project stmodel-guangzhou \
    --experiment-name ${EXPERIMENT_NAME} \
    --spatial-weight 0.9 \
    --temporal-weight 0.3 \
    --temporal-soft-weight 0.2 \
    --spatial-temporal-weight 0.2 \
    --group \
    --group_element 4 \
    --temperature 0.07 \
    --n-views 2 \
    --warmup-epochs 20 \
    --log-every-n-steps 100 \
    --cuda \
    --labels_str "['dense houses', 'trees', 'grass', 'river', 'barren land', 'sidewalks', 'farmland', 'tall buildings', 'soil', 'crop fields', 'roads']" \
    --MODE group-spatial_temporal \
    --log_tag ddp_${NUM_GPUS}gpu \
    --epsilon 0.1

echo "======================================"
echo "🎉 DDP Training completed!"
echo "📁 Logs saved in: /hpc2hdd/home/qwang650/project/yutianjiang/stmodel/training_results/${EXPERIMENT_NAME}"
echo "======================================"