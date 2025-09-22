#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python main_linprobe.py \
--model softcon \
--checkpoint /hpc2hdd/home/qwang650/project/yutianjiang/stmodel/weights/softcon/B13_rn50_softcon.pth \
--wandb linprobe \
--dataset_type indicator \
--train_dataset_ratio 0.8 \
--indicator gdp \
--city Guangzhou
