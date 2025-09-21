import logging
import torch
import os
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

from models.simclr_resnet import ResNetSimCLR
from utils.utils import accuracy, save_checkpoint, save_config_file
from utils.loss import ContrastiveLoss
from utils.get_label import get_label_emb

torch.manual_seed(0)


class SimCLRSpatilTemporal(object):
    
    """
    anchors: [batch_size, group_element, n_views, C, H, W]
    temporal_negatives: [batch_size, group_element, num_neg, C, H, W]
    """
    
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.dataset = kwargs.get('dataset', None)
        
        # 1. 自动创建日志和检查点目录
        # 日志和检查点都将保存在 self.args.log_dir 指定的路径下
        self.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
        os.makedirs(self.args.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 2. 初始化 wandb
        # Set wandb API key and login
        os.environ["WANDB_API_KEY"] = "8f2fc4596fca723c90b10053eeaead122e46603a"
        wandb.login()
        
        wandb.init(
            project=self.args.wandb_project,
            name=self.args.experiment_name,
            config=vars(self.args),
            mode="online"
        )
        
        # 3. 初始化日志记录器 (logging)
        logging.basicConfig(
            filename=os.path.join(self.args.log_dir, 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # 4. 初始化损失函数和工具
        self.Contrastive_loss = ContrastiveLoss(self.args)
        self.temporal_loss = self.Contrastive_loss.temporal_loss
        self.temporal_soft_loss = self.Contrastive_loss.temporal_soft_loss
        self.spatial_group_smoothing_loss = self.Contrastive_loss.spatial_group_smoothing_loss
        self.group_spatial_temporal_loss = self.Contrastive_loss.group_spatial_temporal_loss
        self.accuracy = accuracy

        self.label_emb = get_label_emb(labels_str=self.args.labels_str)

    
    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.mixed_precision)
        
        save_config_file(self.args.log_dir, self.args) 
        
        n_iter = 0
        logging.info(f"Start SimCLRTemporalSpatial training for {self.args.epochs} epochs.")
        logging.info(f"Training on device: {self.args.device}.")
        logging.info(f"Logs and checkpoints will be saved in: {self.args.log_dir}")

        label_emb = self.label_emb.to(self.args.device)

        for epoch in range(self.args.epochs):
            train_loader.dataset.current_epoch = epoch
            self.model.train()
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}"):
                anchors = batch["anchors"].to(self.args.device)
                temporal_negatives = batch["temporal_negatives"].to(self.args.device)
                temporal_negatives_ratios = batch["temporal_negatives_ratios"].to(self.args.device)

                batch_size, group_element, n_views, C, H, W = anchors.shape
                _, _, num_neg, _, _, _ = temporal_negatives.shape

                self.optimizer.zero_grad()

                with autocast(enabled=self.args.mixed_precision):
                    # Reshape and get features
                    anchors = anchors.view(-1, C, H, W)
                    features = self.model(anchors)
                    features_dim = features.shape[-1]
                    features = features.view(batch_size * group_element, n_views, features_dim)

                    temporal_negatives = temporal_negatives.view(-1, C, H, W)
                    temporal_negatives_features = self.model(temporal_negatives).view(
                        batch_size * group_element, num_neg, features_dim
                    )

                    negative_ratios = temporal_negatives_ratios.view(batch_size * group_element, num_neg, -1)

                    # Calculate losses
                    spatial_loss = self.spatial_group_smoothing_loss(features)
                    temporal_logits, temporal_labels_ce, temporal_loss = self.temporal_loss(
                        features=features,
                        temporal_negatives=temporal_negatives_features
                    )
                    
                    anchors_single = features[:, 0, :]
                    _, _, soft_loss, indices, _ = self.temporal_soft_loss(
                        anchors=anchors_single,
                        temporal_negatives=temporal_negatives_features,
                        negative_ratios=negative_ratios,
                        label_emb=label_emb
                    )

                    group_st_loss = self.group_spatial_temporal_loss(
                        anchors=anchors_single,
                        temporal_negatives=temporal_negatives_features,
                        indices=indices
                    )

                    # Combine losses
                    loss = 0.9 * spatial_loss + 0.2 * temporal_loss + 0.3 * group_st_loss
                    loss_2 = 0.2 * soft_loss
                    total_loss = loss + loss_2

                # Backward pass and optimization
                scaler.scale(loss_2).backward(retain_graph=True)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                # Log metrics every N iterations
                if n_iter % self.args.log_every_n_steps == 0:
                    top1_acc, top5_acc = (0, 0)
                    if temporal_logits is not None and temporal_labels_ce is not None:
                        top1_acc, top5_acc = self.accuracy(temporal_logits, temporal_labels_ce, topk=(1, 5))
                        top1_acc = top1_acc.item()
                        top5_acc = top5_acc.item()

                    iter_metrics = {
                        'iter/spatial_loss': spatial_loss.item(),
                        'iter/temporal_loss': temporal_loss.item(),
                        'iter/soft_loss': soft_loss.item(),
                        'iter/group_st_loss': group_st_loss.item(),
                        'iter/total_loss': total_loss.item(),
                        'iter/top1_accuracy': top1_acc,
                        'iter/top5_accuracy': top5_acc,
                        'iter/learning_rate': self.optimizer.param_groups[0]['lr']
                    }
                    wandb.log(iter_metrics, step=n_iter)

                n_iter += 1

            # --- End of Epoch ---
            
            # Adjust learning rate
            if epoch >= self.args.warmup_epochs:
                self.scheduler.step()

            # Log epoch-level summary
            top1_acc, top5_acc = (0, 0)
            if temporal_logits is not None and temporal_labels_ce is not None:
                top1_acc, top5_acc = self.accuracy(temporal_logits, temporal_labels_ce, topk=(1, 5))
                top1_acc = top1_acc.item()
                top5_acc = top5_acc.item()

            epoch_metrics = {
                'epoch/spatial_loss': spatial_loss.item(),
                'epoch/temporal_loss': temporal_loss.item(),
                'epoch/soft_loss': soft_loss.item(),
                'epoch/group_st_loss': group_st_loss.item(),
                'epoch/total_loss': total_loss.item(),
                'epoch/top1_accuracy': top1_acc,
                'epoch/top5_accuracy': top5_acc,
                'epoch/learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_num': epoch + 1
            }
            # 使用 n_iter 作为 step 来确保单调递增
            wandb.log(epoch_metrics, step=n_iter)
            
            log_message = (f"Epoch {epoch+1}/{self.args.epochs} Summary: "
                           f"Total Loss={total_loss.item():.4f}, "
                           f"Top1 Acc={top1_acc:.4f}, "
                           f"LR={self.optimizer.param_groups[0]['lr']:.6f}")
            logging.info(log_message)
            
            # Save checkpoint every 50 epochs
            if (epoch + 1) % 1 == 0 or (epoch + 1) == self.args.epochs:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1:04d}.pth')
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                }, is_best=False, filename=checkpoint_path)
                
                logging.info(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
                # wandb.save(checkpoint_path) # Optional: save checkpoint to wandb cloud

        # --- End of Training ---
        
        logging.info("Training has finished.")
        # Save final model
        final_checkpoint_path = os.path.join(self.checkpoint_dir, 'final_model.pth')
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }, is_best=False, filename=final_checkpoint_path)
        
        logging.info(f"Final model saved to {final_checkpoint_path}")
        # wandb.save(final_checkpoint_path) # Optional: save final model to wandb cloud
        
        wandb.finish()