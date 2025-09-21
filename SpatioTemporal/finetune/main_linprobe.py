import argparse
import torch
import torch.nn as nn
import wandb
import sys
from models.simclr_resnet import ResNetSimCLR
from models.softcon import SoftCon, load_checkpoint_rgb
import os


from ft_datasets.Eurosat import build_eurosat_dataset
from engine_finetune import train_one_epoch, evaluate

# export WANDB_API_KEY=your_api_key_here 


def get_args_parser():
    parser = argparse.ArgumentParser('Linear probing for ResNetSimCLR', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--model', default='resnetsimclr', type=str, choices=['softcon', 'resnetsimclr'])
    parser.add_argument('--base_model', default='resnet50', type=str)
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to SimCLR checkpoint')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--wandb', type=str, default='linprobe', help='Wandb project name')
    parser.add_argument('--dataset_type', default='eurosat', type=str)
    return parser


def main():
    args = get_args_parser().parse_args()
    device = torch.device(args.device)


    # api_key = os.environ.get("8f2fc4596fca723c90b10053eeaead122e46603a")
    # wandb.login(key=api_key, relogin=True)

    # Build datasets
    dataset_train = build_eurosat_dataset(is_train=True, args=args)
    dataset_val = build_eurosat_dataset(is_train=False, args=args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    # Load model
    if args.model == 'resnetsimclr':
        # Create model with original architecture
        model = ResNetSimCLR(base_model=args.base_model, out_dim=128)

        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['state_dict']

        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

        # Freeze all parameters for pure linear probing
        for param in model.parameters():
            param.requires_grad = False

        # Remove last layer of projection and add classification head
        model.projection = model.projection[:3]  # Keep first two layers: Linear-ReLU-Linear
        feature_dim = model.projection[2].out_features

        head = nn.Linear(feature_dim, 10)  

    elif args.model == 'softcon':
        model = SoftCon(base_model=args.base_model)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint
        missing_keys, unexpected_keys = load_checkpoint_rgb(model, args.checkpoint)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

        # Freeze all parameters for pure linear probing
        for param in model.parameters():
            param.requires_grad = False

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            feature_dim = model(dummy_input).shape[1]
        head = nn.Linear(feature_dim, 10)  

    else:
        raise NotImplementedError(f"Model {args.model} not supported")


    model.to(device)
    head.to(device)
    # model.eval()

    # Optimizer only for classification head
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Initialize wandb
    if args.wandb:
        wandb.init(project=args.wandb)
        wandb.config.update(args)

    print(f"Start linear probing for {args.epochs} epochs")
    best_acc = 0.0

    for epoch in range(args.epochs):
        # Train
        train_stats = train_one_epoch(
            model, head, criterion, data_loader_train,
            optimizer, device, epoch, args, model_freeze=True
        )

        # Evaluate
        test_stats = evaluate(
            data_loader_val, model, head, device, criterion, args
        )

        acc = test_stats['acc1'] * 100
        print(f"Epoch {epoch}: Train Loss = {train_stats['loss']:.4f}, Test Acc = {acc:.2f}%")

        best_acc = max(best_acc, acc)

        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_stats['loss'],
                'test_acc': acc,
                'best_acc': best_acc
            })

    print(f"Best accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()