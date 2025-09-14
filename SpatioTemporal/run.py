import os
import argparse
from posixpath import isfile
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import wandb

from datasets.spatio_temporal_dataset import SpatioTemporalDataset
from datasets.contrastive_learning_datasets import ContrastiveLearningDataset    
from utils.collate_fn import custom_collate_fn
from models.simclr_resnet import ResNetSimCLR
from simclr import SimCLRSpatilTemporal


def main():
    parser = argparse.ArgumentParser(description='PyTorch SimCLR with Spatio-Temporal Contrastive Learning')   
    
    # 数据集参数
    parser.add_argument('--data', default='/data/yutianjiang/multi_temporal/Guangzhou/Data_temp/jpg', 
                        help='path to dataset root folder')
    parser.add_argument('--dataset-name', default='STlabelScoreGroupDataset', 
                        help='dataset name', choices=['stl10', 'cifar10', 
                                                        'SpatioTemporalDataset',
                                                        'STlabelpositive',   
                                                        'STlabel',   
                                                        'STlabel_spatio_same', 
                                                        'STlabelScoreGroupDataset'  # 修正名称
                                                        ])
    
    # 模型参数
    parser.add_argument('--arch', default='resnet50', choices=['resnet18', 'resnet50'],
                        help='model architecture')
    parser.add_argument('--out_dim', default=128, type=int, 
                        help='feature dimension (default: 128)')
    
    # 训练参数
    parser.add_argument('--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', 
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N',  
                        help='mini-batch size (group_num)')
    parser.add_argument('--lr', default=0.0003, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W',    
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--seed', default=42, type=int, 
                        help='seed for initializing training.')
    
    # 对比学习参数
    parser.add_argument('--lambda_temporal', default=0.5, type=float, 
                        help='Weight for temporal contrastive loss')
    parser.add_argument('--num_neg', default=20, type=int, 
                        help='Number of temporal negative samples')
    parser.add_argument('--temperature', default=0.07, type=float, 
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--n-views', default=2, type=int, metavar='N', 
                        help='Number of views for contrastive learning training.')
    
    # 日志和保存参数
    parser.add_argument('--log_root', default=
                        '/data/yutianjiang/training_results/spatio_temporal/Guangzhou/Ablation/ablation_debug/pretrain', 
                        help='Root directory for logging')
    parser.add_argument('--log-every-n-steps', default=100, type=int, 
                        help='Log every n steps')
    parser.add_argument('--warmup-epochs', default=20, type=int,
                        help='Number of warmup epochs before applying scheduler')
    
    # Label和模式参数
    parser.add_argument('--labels_str', default="['dense houses', 'trees', 'grass', 'river', 'barren land', 'sidewalks', 'farmland', 'tall buildings', 'soil', 'crop fields', 'roads']", 
                        type=str, help='Labels string for embedding')
    parser.add_argument('--MODE', default='group-spatial_temporal', type=str,
                        help='Training mode')
    parser.add_argument('--log_tag', default='debug', type=str,
                        help='Log tag for experiment identification')
    
    # Group参数
    parser.add_argument('--group_num', default=4, type=int,
                        help='Number of groups (should match batch_size)')
    parser.add_argument('--group_element', default=4, type=int,
                        help='Number of elements per group')
    parser.add_argument('--group', default=True, action='store_true',
                        help='Use group loss functions')
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help='Epsilon for label smoothing in spatial group loss')
    
    # 设备参数
    parser.add_argument('--cuda', default=True, action='store_true',
                        help='Use cuda for training')
    parser.add_argument('--mixed-precision', default=False, action='store_true',
                        help='Use mixed precision training')
    
    # Wandb 参数
    parser.add_argument('--wandb-project', default='stmodel-simclr', type=str,
                        help='Wandb project name')
    parser.add_argument('--experiment-name', default=None, type=str,
                        help='Experiment name for wandb')
    
    # Loss 权重参数
    parser.add_argument('--spatial-weight', default=0.3, type=float,
                        help='Weight for spatial group smoothing loss')
    parser.add_argument('--temporal-weight', default=0.3, type=float,
                        help='Weight for temporal loss')
    parser.add_argument('--temporal-soft-weight', default=0.2, type=float,
                        help='Weight for temporal soft loss')
    parser.add_argument('--spatial-temporal-weight', default=0.2, type=float,
                        help='Weight for group spatial-temporal loss')
    
    # 断点续训
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--checkpoint-path', default='', type=str,
                        help='path to checkpoint for resume')
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        import numpy as np
        import random
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    # 验证group参数 - batch_size就是group_num
    if args.group:
        # 确保batch_size就是group_num
        args.group_num = args.batch_size
        print(f"Group mode: group_num={args.group_num}, group_element={args.group_element}")
        print(f"Effective batch size: {args.group_num * args.group_element}")
    
    # Wandb 实验名称设置
    if args.experiment_name is None:
        current_time = datetime.now().strftime('%Y%m%d_%H%M')
        args.experiment_name = f"{args.MODE}_{args.log_tag}_{current_time}"
    
    # 保存路径设置
    log_dir = os.path.join(args.log_root, args.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    
    # 设备设置
    if torch.cuda.is_available() and args.cuda:
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
        print('========== Using GPU ==========')
    else: 
        args.device = torch.device('cpu')
        print('========== Using CPU ==========')
        
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    
    # Resume处理
    start_epoch = 0
    wandb_run_id = None
    if args.resume and args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        start_epoch = checkpoint['epoch']
        if 'wandb_run_id' in checkpoint:
            wandb_run_id = checkpoint['wandb_run_id']
        print(f"Resuming from epoch {start_epoch}")
    
    # 准备数据集
    dataset = ContrastiveLearningDataset(
        args.data, 
        args.num_neg, 
        max_epoch=args.epochs, 
        current_epoch=start_epoch
    )
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
    
    # DataLoader - batch_size就是group_num
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,  # 这里就是group_num
        shuffle=True, 
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    # 模型创建
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    
    # 优化器和调度器
    optimizer = torch.optim.Adam(
        model.parameters(), 
        args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(train_loader) * args.epochs, 
        eta_min=0, 
        last_epoch=-1
    )

    # Resume模型状态
    if args.resume and args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
    
    # 设置wandb resume
    if wandb_run_id:
        os.environ['WANDB_RUN_ID'] = wandb_run_id
        os.environ['WANDB_RESUME'] = 'must'
    
    # 创建SimCLR训练器
    simclr = SimCLRSpatilTemporal(
        args=args, 
        model=model, 
        optimizer=optimizer, 
        scheduler=scheduler,
        dataset=dataset
    )
    
    # 开始训练
    simclr.train(train_loader)
    print(f"Training completed. Logs are saved in {args.log_dir}")
    
    
if __name__ == "__main__":
    main()