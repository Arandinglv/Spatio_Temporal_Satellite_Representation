#!/bin/bash

# DDP训练脚本 - 多卡时空对比学习 (SOTA级别实现)
# 使用方法:
#   bash train_ddp.sh                    # 使用默认GPU 0,1,2,3
#   bash train_ddp.sh "0,1,2,3"         # 指定特定GPU
#   bash train_ddp.sh "4,5,6,7"         # 使用其他GPU

# GPU配置 (可通过第一个参数指定)
GPUS=${1:-"0,1"}  # 默认使用GPU 0,1,2,3
export CUDA_VISIBLE_DEVICES=${GPUS}
NUM_GPUS=$(echo ${GPUS} | tr ',' '\n' | wc -l)

# 基础参数 (使用与原脚本相同的路径和参数)
DATA_PATH="/hpc2hdd/home/qwang650/project/yutianjiang/stmodel/dataset/Guangzhou"
LOG_ROOT="/hpc2hdd/home/qwang650/project/yutianjiang/stmodel/training_results"
DATASET_NAME="STlabelScoreGroupDataset"
ARCH="resnet50"
BATCH_SIZE=20  # 每个GPU的batch size (与原脚本一致)
EPOCHS=100
LR=0.0003
WEIGHT_DECAY=1e-4
OUT_DIM=128

# 实验配置
MODE="group-spatial_temporal"
LOG_TAG="ddp_4gpu"
GROUP_ELEMENT=4
GROUP_NUM=${BATCH_SIZE}  # group_num等于batch_size

# 损失权重 (与原脚本一致)
SPATIAL_WEIGHT=0.3
TEMPORAL_WEIGHT=0.3
TEMPORAL_SOFT_WEIGHT=0.2
SPATIAL_TEMPORAL_WEIGHT=0.2

# 对比学习参数 (与原脚本一致)
NUM_NEG=6  # 与原脚本一致
TEMPERATURE=0.07
N_VIEWS=2
EPSILON=0.1

# 训练参数
WORKERS=12
WARMUP_EPOCHS=20
LOG_EVERY_N_STEPS=100
SEED=42

# Wandb配置
WANDB_PROJECT="stmodel-guangzhou"

# Labels配置 (与原脚本一致)
LABELS_STR="['dense houses', 'trees', 'grass', 'river', 'barren land', 'sidewalks', 'farmland', 'tall buildings', 'soil', 'crop fields', 'roads']"

# 创建实验名称
EXPERIMENT_NAME="guangzhou_spatiotemporal_ddp_$(date +%Y%m%d_%H%M)"

# SOTA级别的验证和监控
echo "======================================"
echo "📊 DDP Training Configuration (SOTA)"
echo "======================================"
echo "🔥 GPUs: ${GPUS} (${NUM_GPUS} cards)"
echo "📁 Experiment: ${EXPERIMENT_NAME}"
echo "💾 Data: ${DATA_PATH}"
echo "📈 Total effective batch size: $((BATCH_SIZE * NUM_GPUS * GROUP_ELEMENT * N_VIEWS))"
echo "🎯 Per GPU batch size: ${BATCH_SIZE}"
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
if [ ! -d "${DATA_PATH}" ]; then
    echo "❌ ERROR: Data path not found: ${DATA_PATH}"
    exit 1
fi

echo "✅ All checks passed. Starting training..."

# 使用torchrun启动DDP训练
torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    run.py \
    --data ${DATA_PATH} \
    --dataset-name ${DATASET_NAME} \
    --arch ${ARCH} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --out_dim ${OUT_DIM} \
    --log_root ${LOG_ROOT} \
    --experiment-name ${EXPERIMENT_NAME} \
    --MODE ${MODE} \
    --log_tag ${LOG_TAG} \
    --group_num ${GROUP_NUM} \
    --group_element ${GROUP_ELEMENT} \
    --spatial-weight ${SPATIAL_WEIGHT} \
    --temporal-weight ${TEMPORAL_WEIGHT} \
    --temporal-soft-weight ${TEMPORAL_SOFT_WEIGHT} \
    --spatial-temporal-weight ${SPATIAL_TEMPORAL_WEIGHT} \
    --lambda_temporal ${LAMBDA_TEMPORAL} \
    --num_neg ${NUM_NEG} \
    --temperature ${TEMPERATURE} \
    --n-views ${N_VIEWS} \
    --wandb-project ${WANDB_PROJECT} \
    --world-size ${NUM_GPUS} \
    --workers 8 \
    --seed 42 \
    --mixed-precision \
    --cuda

echo "======================================"
echo "DDP Training completed!"
echo "Logs saved in: ${LOG_ROOT}/${EXPERIMENT_NAME}"
echo "======================================"