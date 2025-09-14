import os
import torch 
import torch.nn as nn   
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
import numpy as np
import wandb

from utils.crop import RandomResizedCrop
from models.simclr_resnet import ResNetSimCLR
from engine_finetune import evaluate, train_one_epoch
from downstream_datasets import create_dataset
from datasets.datasets_inference import BigEarthNet


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/yutianjiang/datasets/Eurosat_RGB')
    parser.add_argument('--dataset_type', type=str, default='eurosat', choices=['eurosat', 'bigearthnet', 'indicator'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--checkpoint', type=str, 
                        default='/data/yutianjiang/training_results/spatio_temporal/vanilla/Guangzhou/label_soft_all/label_soft_pretrain_on_ST/resnet18.pth.tar')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--classes_num', type=int, default=10)
    parser.add_argument('--base_model', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--accum_iter', type=int, default=1)
    parser.add_argument('--indicator', type=str, default=None, choices=["carbon", "population", "gdp"])
    parser.add_argument('--indicator_city', type=str, default=None, choices=["Guangzhou"])
    parser.add_argument('--train_dataset_ratio', type=float, default=0.8)
    
    # Wandb parameters
    parser.add_argument('--project_name', type=str, default=None, help="Wandb project name")
    parser.add_argument('--experiment_name', type=str, default=None, help="Wandb experiment name")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    
    # Initialize wandb
    if args.project_name:
        exp_name = args.experiment_name or f"{args.dataset_type}_{args.base_model}_linprobe"
        wandb.init(project=args.project_name, name=exp_name, config=args)
        args.wandb = args.project_name  # For engine compatibility
    
    # Data transforms
    transform_train = transforms.Compose([
        RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # Load datasets
    if args.dataset_type == 'eurosat':
        train_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=transform_train)
        val_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'val'), transform=transform_val)
    elif args.dataset_type == 'bigearthnet':
        train_dataset = BigEarthNet(root_folder=args.data_dir, transform=transform_train, type='train')
        val_dataset = BigEarthNet(root_folder=args.data_dir, transform=transform_val, type='val')
    elif args.dataset_type == 'indicator':
        train_dataset, val_dataset, _, _, _ = create_dataset.create_indicator_datasets(
            args=args, transform_train=transform_train, transform_val=transform_val)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Load model
    model = ResNetSimCLR(base_model=args.base_model)
    model.to(args.device)
    
    # Load pretrained weights
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("Loading pretrained weights:", msg)
    
    # Linear probing setup
    print("=> Linear probing mode")
    for _, param in model.named_parameters():
        param.requires_grad = False
    model.projection = nn.Identity()
    
    # Get encoder output dimension
    with torch.no_grad():
        dummy_input = torch.zeros(1, 3, 224, 224).to(args.device)
        features = model(dummy_input)
        feature_dim = features.shape[1]
    print(f"Encoder output dimension: {feature_dim}")
    
    # Create classification head
    head = nn.Sequential(
        nn.BatchNorm1d(feature_dim, affine=False, eps=1e-6),
        nn.Linear(feature_dim, args.classes_num)
    )
    head.to(args.device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    
    # Loss function
    if args.dataset_type == 'eurosat':
        criterion = nn.CrossEntropyLoss()
    elif args.dataset_type == 'bigearthnet':
        criterion = nn.BCEWithLogitsLoss()
    elif args.dataset_type == 'indicator':
        criterion = nn.MSELoss()
    
    print(f"Start training for {args.epochs} epochs")
    best_acc1 = 0.0
    best_mse = 1e6
    
    # Training loop
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(model, head, criterion, train_loader, optimizer, args.device, epoch, args, model_freeze=True)
        
        if epoch >= args.warmup_epochs:
            lr_scheduler.step()

        val_stats = evaluate(val_loader, model, head, args.device, criterion, args)
        
        # Log and track best
        if args.dataset_type == 'indicator':
            print(f"[Epoch {epoch:3d}] Train Loss: {train_stats['loss']:.4f}, Val MSE: {val_stats['mse']:.4f}, R²: {val_stats['r2']:.4f}")
            
            if args.project_name:
                wandb.log({
                    'epoch': epoch, 'train_loss': train_stats['loss'], 'val_loss': val_stats['loss'],
                    'val_mse': val_stats['mse'], 'val_rmse': val_stats['rmse'], 
                    'val_mae': val_stats['mae'], 'val_r2': val_stats['r2']
                })
            
            if val_stats['mse'] < best_mse:
                best_mse = val_stats['mse']
        else:
            print(f"[Epoch {epoch:3d}] Train Loss: {train_stats['loss']:.4f}, Val Acc@1: {val_stats['acc1']*100:.2f}%")
            
            if args.project_name:
                wandb.log({
                    'epoch': epoch, 'train_loss': train_stats['loss'], 'val_loss': val_stats['loss'],
                    'val_acc1': val_stats['acc1'], 'val_acc5': val_stats['acc5']
                })
            
            if val_stats['acc1'] > best_acc1:
                best_acc1 = val_stats['acc1']
    
    # Final results
    if args.dataset_type == 'indicator':
        print(f"Training completed. Best val MSE = {best_mse:.4f}")
        if args.project_name:
            wandb.log({'best_val_mse': best_mse})
    else:
        print(f"Training completed. Best val acc@1 = {best_acc1*100:.2f}%")
        if args.project_name:
            wandb.log({'best_val_acc1': best_acc1})
    
    if args.project_name:
        wandb.finish()


if __name__ == "__main__":
    main()