#!/bin/bash

# 选择CUDA设备
export CUDA_VISIBLE_DEVICES=2

# 训练命令
python3 run.py \
    --data /data/yutianjiang/multi_temporal/Guangzhou/Data_temp/jpg \
    --dataset-name SpatioTemporalDataset\
    --arch resnet18 \
    --workers 12 \
    --epochs 500 \
    --batch-size 64 \
    --lr 0.0003 \
    --weight-decay 1e-4 \
    --seed 42 \
    --lambda_temporal 0.5 \
    --num_neg 20 \
    --log_root /data/yutianjiang/training_results/spatio_temporal/vanilla/Guangzhou \
    --out_dim 128 \
    --log-every-n-steps 100 \
    --temperature 0.07 \
    --n-views 2 \
    --warmup-epoch 100 \
    --description "Nolabel"
