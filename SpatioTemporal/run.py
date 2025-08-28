import os
import argparse
from posixpath import isfile
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

from datasets.spatio_temporal_dataset import SpatioTemporalDataset
from datasets.contrastive_learning_datasets import ContrastiveLearningDataset    
from utils.collate_fn import custom_collate_fn
from models.simclr_resnet import ResNetSimCLR
from simclr import SimCLRSpatilTemporal

from warmup_scheduler import GradualWarmupScheduler

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets.batch_sampler import YearBatchSampler

# TODO:设置RESUME 断点续训

# 最后parser中的的连词符都会变成下划线
def main():
    parser = argparse.ArgumentParser(description='PyTorch SimCLR with Spatio-Temporal Contrastive Learning')   
    # 数据集位置 
    parser.add_argument('--data', default='/data/yutianjiang/multi_temporal/Shanghai/Data_temp/jpg', 
                        help='path to dataset root folder')
    # 数据集名称, 按照ContrastiveLearningDataset中的标准
    parser.add_argument('--dataset-name', default='SpatioTemporalDataset', 
                        help='dataset name', choices=['stl10', 'cifar10', 
                                                        'SpatioTemporalDataset',  # 不带label的原生数据集
                                                        'STlabelpositive',   
                                                        # 带label, 相同label不同年作为正样本(对于三个loss都是)
                                                        'STlabel',   
                                                        # 带label, positive是自己的transform, 
                                                        'STlabel_spatio_same', 
                                                        # 带label, positive是自己的transform, 同一个batchsize内的anchor样本来自于同一年
                                                        ])
    # 模型名字
    parser.add_argument('--arch', default='resnet18', choices=['resnet18', 'resnet50'],
                        help='model architecture: resnet18 or resnet50 (default: resnet18)')
    # workers数量
    parser.add_argument('--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    # 迭代次数
    parser.add_argument('--epochs', default=200, type=int, metavar='N', 
                        help='number of total epochs to run')
    # batch-size
    parser.add_argument('--batch-size', default=16, type=int, metavar='N',  
                        help='mini-batch size (default: 128)')
    # learning_rate
    parser.add_argument('--lr', default=0.0003, type=float, metavar='LR',
                        help='initial learning rate')
    # weight_decay
    parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W',    
                        help='weight decay (default: 1e-4)')
    # 随机数种子
    parser.add_argument('--seed', default=None, type=int, 
                        help='seed for initializing training.')
    # 是否禁用cuda
    parser.add_argument('--cuda', default=True, action='store_true',
                        help='Use cuda for training')  
    # 默认为True, 命令行中出现--disable-cuda时, 会禁用cuda
    
    # # 用几号卡
    # parser.add_argument('--gpu-index', default=2, type=int, help='GPU Index')
    # 时空对比学习损失函数的权重
    parser.add_argument('--lambda_temporal', default=1.0, type=float, 
                        help='Weight for temporal contrastive loss')
    # # 时间负样本每个年份增强的数量
    # parser.add_argument('--num_tempo_neg_aug', default=10, type=int, 
    #                     help='Number of temporal negative samples')
    
    # 时间负样本数量
    parser.add_argument('--num_neg', default=20, type=int, 
                        help='Number of temporal negative samples')
    # log保存位置
    parser.add_argument('--log_root', default=
                        '/data/yutianjiang/training_results/simclr/2010_2019', 
                        help='Root directory for logging')
    # 混精训练
    parser.add_argument('--mixed-precision', default=True, action='store_true',
                        help='Whether or not to use 16-bit and 32-nit precision GPU training. \
                            默认设置为True, 如果在脚本中指定--mixed-precision, 则为false')
    # projection_head的维度大小
    parser.add_argument('--out_dim', default=128, type=int, help='feature dimension (default: 128)')
    # 记录, 不变
    parser.add_argument('--log-every-n-steps', default=100, type=int, 
                        help='Log every n steps')
    # softmax温度系数, 不变
    parser.add_argument('--temperature', default=0.07, type=float, 
                        help='softmax temperature (default: 0.07)')
    # 视图数, 不变
    parser.add_argument('--n-views', default=2, type=int, metavar='N', 
                        help='Number of views for contrastive learning training.')
    # warmup
    parser.add_argument('--warmup-epochs', default=0, type=int,
                        help='Number of warmup epochs before applying scheduler')
    
    # 断点续训
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    # checkpoint保存地址
    # TODO: 保存在log_root下方
    parser.add_argument('--checkpoint-path', default='/data/yutianjiang/training_results',
                        type=str, metavar='PATH', help='path to save checkpoint')
    parser.add_argument('--description', default='simclr', type=str,
                        help='Description of the experiment, as the name of the log folder')
    
    # batchsize内的区域样本是同一个年份
    parser.add_argument('--same-year', default=False, action='store_true',
                        help='Whether or not to sample from the same year in the batch')
    
    args = parser.parse_args()
    
    # ==================== RESUME ====================
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path)
            srart_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"Running from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.checkpoint_path}")
    
    # =================================================
    
    assert args.n_views == 2, \
        "Only two view training is supported. Please use --n-views 2."
    
    # 是否使用gpu
    if torch.cuda.is_available() and args.cuda:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        args.device = torch.device('cuda')
        cudnn.deterministic = True  # 提高实验的可重复性
        cudnn.benchmark = True
        print('========== Using GPU ==========')
    else: 
        args.device = torch.device('cpu')
        args.gpu_index = -1
        print('========== Using CPU ==========')
        

        
    
    # 准备数据集
    dataset = ContrastiveLearningDataset(args.data, args.num_neg)
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
    # custom_dataloder = YearBatchSampler(
    #     dataset=train_dataset,
    #     possible_years=train_dataset.years,
    #     batch_size=args.batch_size,
    #     drop_last=True
    # )
    custom_dataloder = YearBatchSampler(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    if args.same_year:
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_sampler=custom_dataloder,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size = args.batch_size,
            shuffle=True, 
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,  # 丢弃最后一个不完整的批次
            collate_fn=custom_collate_fn
        )
        
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    
    # TODO: 先用linear scale rule预热然后余弦退火
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=len(train_loader)*args.epochs, eta_min=0, last_epoch=-1)
    
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, 
        T_max=len(train_loader) * (args.epochs - args.warmup_epochs),
        eta_min=0, 
        last_epoch=-1
    )
    
    scheduler = GradualWarmupScheduler(
        optimizer,  
        multiplier=1,   
        total_epoch=args.warmup_epochs,    
        after_scheduler=cosine_scheduler   
    )

    current_time = datetime.now().strftime('%Y%m%d_%H%M')
    # log_dir = os.path.join(args.log_root, f"{current_time}_")
    log_dir = os.path.join(args.log_root, args.description, current_time)
    os.makedirs(log_dir, exist_ok=True) if not os.path.exists(log_dir) else None
    
    args.log_dir = log_dir
    
    
    simclr = SimCLRSpatilTemporal(
        args=args, 
        model=model, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        # start_epoch=start_epoch
        )
    simclr.train(train_loader)
    print(f"Training completed. Checkpoints and logs are saved in {log_dir}")
    
    
if __name__ == "__main__":
    main()
