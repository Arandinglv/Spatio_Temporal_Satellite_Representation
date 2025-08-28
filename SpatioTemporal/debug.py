# import logging
# import torch
# import torch.nn.functional as F
# from datetime import datetime
# import os
# from torch.cuda.amp import GradScaler, autocast
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# # from utils import save_config_file, accuracy, save_checkpoint

# from models.simclr_resnet import ResNetSimCLR
# from utils.utils import spatial_accuracy, temporal_accuracy_neg, \
#     temporal_accuracy_pos_neg, save_checkpoint, save_config_file
    
# from utils.loss import ContrastiveLoss

# from datasets.spatio_temporal_dataset import SpatioTemporalDataset
# from datasets.contrastive_learning_datasets import ContrastiveLearningDataset  
# from utils.collate_fn import custom_collate_fn
# from warmup_scheduler import GradualWarmupScheduler


# torch.manual_seed(0)


# dataset = ContrastiveLearningDataset('/data/yutianjiang/multi_temporal/Shanghai/Data_temp/jpg')
# train_dataset = dataset.get_dataset('SpatioTemporalDataset', n_views=2)

# train_loader = torch.utils.data.DataLoader(
#     train_dataset, 
#     batch_size=64, 
#     shuffle=True,
#     num_workers=12,
#     pin_memory=True,
#         drop_last=True,  # 丢弃最后一个不完整的批次
#         collate_fn=custom_collate_fn
# )

# for idx, data in enumerate(train_loader):
#     anchors = data['anchors']
#     temporal_negatives = data['temporal_negatives']
#     print(anchors.shape)
#     print('===========')
#     print(temporal_negatives.shape)

#     break

# # import torch

# # a = torch.arange(16).reshape((4, 2, 2))
# # print(a)
# # print('==========')
# # print(a.view(-1, 2))


# import torch

# a = torch.arange(6).reshape(2, 1, 3)
# b = torch.arange(12).reshape(2, 2, 3)
# c = torch.sum(a * b, dim=-1)
# d = torch.matmul(a, b.transpose(1, 2))
# print(c.shape)
# print(d.shape)


# import os
# import shutil
# import re

# # 设置源文件夹（包含2017年数据）和目标文件夹（包含2013年数据）
# source_folder = '/data/yutianjiang/multi_temporal/image_2017'
# target_folder = '/data/yutianjiang/multi_temporal/Guangzhou/Data_temp/jpg/2010'
# destination_folder = '/data/yutianjiang/multi_temporal/image_2010'

# # 创建目标文件夹 (如果不存在的话)
# if not os.path.exists(destination_folder):
#     os.makedirs(destination_folder)

# # 获取源文件夹中的所有图像文件
# source_files = os.listdir(source_folder)

# # 正则表达式用来从文件名中提取经纬度
# pattern = re.compile(r'Guangzhou_(\d{4})_(\d+\.\d+)_([+-]?\d+\.\d+)_(\d+\.\d+)_([+-]?\d+\.\d+).jpg')

# # 遍历源文件夹中的每个文件
# for source_file in source_files:
#     # 匹配源文件名的经纬度
#     match = pattern.search(source_file)
#     if match:
#         year = match.group(1)  # 提取年份 (2017)
#         lon1 = match.group(2)  # 提取第一个经度
#         lat1 = match.group(3)  # 提取第一个纬度
#         lon2 = match.group(4)  # 提取第二个经度
#         lat2 = match.group(5)  # 提取第二个纬度
        
#         # 构建2013年版本的图像文件名
#         target_file_name = f"Guangzhou_2010_{lon1}_{lat1}_{lon2}_{lat2}.jpg"
        
#         # 构建目标文件路径
#         target_file = os.path.join(target_folder, target_file_name)
        
#         # 检查目标文件夹中是否有匹配的文件
#         if os.path.exists(target_file):
#             # 构建新的目标文件路径
#             destination_file = os.path.join(destination_folder, target_file_name)
            
#             # 复制文件到新的目录
#             shutil.copy(target_file, destination_file)
#             print(f"已将文件复制到: {destination_file}")
#         else:
#             print(f"没有找到匹配文件: {target_file_name}")
#     else:
#         print(f"无法从文件名中提取经纬度: {source_file}")


# import torch

# a = torch.arange(12).reshape(2, 3, 2)
# print(a.view(-1, 2).shape)
# print(a.shape)

import os

path = '/data/jiabo/samcd/filter_test_all_250-2.5/2010_2011'
files = os.listdir(path)
# 清点文件夹下的文件数量:
print(len(files))
image1 = '/data/yutianjiang/multi_temporal/Guangzhou/Data_temp/jpg/2010/Guangzhou_2010_113.422471_23.156953_113.427352_23.152437.jpg'
image2 = '/data/yutianjiang/multi_temporal/Guangzhou/Data_temp/jpg/2011/Guangzhou_2011_113.422471_23.156953_113.427352_23.152437.jpg'

# 可视化两张图像, 放在一个图框里
# import matplotlib.pyplot as plt
# from skimage.io import imread

# img1 = imread(image1)
# img2 = imread(image2)

# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(img1)
# axes[1].imshow(img2)
# plt.show()

import torch 
import torch.nn.functional as F

a = (torch.arange(24) + 1).reshape((6, 4))
print(a)

b1 = torch.tensor([0, 0.001, 0, 0.002, 0, 0.003]).view(-1, 1)
b2 = torch.tensor([0.001, 0.003, 0.002, 0, 0.004, 0]).view(-1, 1)

b1 = F.normalize(b1, p=2, dim=1)
b2 = F.normalize(b2, p=2, dim=1)

# 广播相乘
a1 = a * b1
a2 = a * b2

# 归一化特征向量
a1 = F.normalize(a1, p=2, dim=1)
a2 = F.normalize(a2, p=2, dim=1)

# 计算余弦相似度矩阵
sim = torch.matmul(a1, a2.T)

sim_2 = torch.matmul(b1, b2.T)

print("相似度矩阵:\n", sim)
print("平均相似度分数:", sim.mean().item())
print("最大相似度分数:", sim.max().item())
print("对角线相似度分数:", sim.diag().mean().item())

print("===========")

print("相似度矩阵:\n", sim_2)
print("平均相似度分数:", sim_2.mean().item())
print("最大相似度分数:", sim_2.max().item())
print("对角线相似度分数:", sim_2.diag().mean().item())
