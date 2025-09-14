#!/bin/bash
# filepath: /hpc2hdd/home/qwang650/project/yutianjiang/stmodel/models/SpatioTemporal/train_stmodel.sh

# 训练spatio-temporal SimCLR模型
# 使用广州数据集

export CUDA_VISIBLE_DEVICES=1

python run.py \
    --data /hpc2hdd/home/qwang650/project/yutianjiang/stmodel/dataset/Guangzhou \
    --dataset-name STlabelScoreGroupDataset \
    --arch resnet50 \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.0003 \
    --weight-decay 1e-4 \
    --seed 42 \
    --num_neg 6 \
    --workers 12 \
    --log_root /hpc2hdd/home/qwang650/project/yutianjiang/stmodel/training_results \
    --wandb-project stmodel-guangzhou \
    --experiment-name guangzhou_spatiotemporal_$(date +%Y%m%d_%H%M) \
    --spatial-weight 0.3 \
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
    --log_tag debug \
    --epsilon 0.1