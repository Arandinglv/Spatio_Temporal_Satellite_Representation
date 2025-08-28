#!/bin/bash

# 选择CUDA设备
export CUDA_VISIBLE_DEVICES=2  # 设置GPU索引为2，您可以根据需要调整此项

# 训练命令
python3 run.py \
    --data /data/yutianjiang/multi_temporal/Shanghai/Data_temp/jpg \
    --dataset-name SpatioTemporalDataset \
    --arch resnet18 \
    --workers 12 \
    --epochs 200 \
    --batch-size 16 \
    --lr 0.0003 \
    --weight-decay 1e-4 \
    --seed 42 \
    --lambda_temporal 0.5 \
    --num_temporal_neg 5 \
    --log_root /data/yutianjiang/training_results/simclr/2010_2019 \
    --out_dim 128 \
    --log-every-n-steps 100 \
    --temperature 0.07 \
    --n-views 2 \
    --warmup-epoch 10 \
    --PosOnly