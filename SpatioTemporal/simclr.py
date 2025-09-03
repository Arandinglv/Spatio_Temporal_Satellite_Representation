import logging
import torch
import torch.nn.functional as F
from datetime import datetime
import os
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
# from utils import save_config_file, accuracy, save_checkpoint

from models.simclr_resnet import ResNetSimCLR
from utils.utils import accuracy, save_checkpoint, save_config_file

from utils.loss import ContrastiveLoss

torch.manual_seed(0)


class SimCLRSpatilTemporal(object):
    
    """
    anchors: [batch_size, n_views, C, H, W]
    temporal_negatives: [batch_size, num_neg, C, H, W]
    """
    
    def __init__(self, *args, **kwargs):
    # 也可以写成:
    # def __init__(self, args, model, optimizer, scheduler):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        
        # Initialize wandb
        wandb.init(
            project=self.args.wandb_project,
            name=self.args.experiment_name,
            config=vars(self.args),
            mode="online"
        )
        
        logging.basicConfig(
            filename=os.path.join(self.args.log_dir, 'training.log'),
            level=logging.DEBUG,
            format='%(asctime)s %(levelname)s:%(message)s'
        )
        self.Contrastive_loss = ContrastiveLoss(self.args)
        # 使用现有的loss函数
        self.spatial_loss = self.Contrastive_loss.spatial_group_smoothing_loss
        self.spatial_temporal_loss = self.Contrastive_loss.group_spatial_temporal_loss
        self.temporal_loss = self.Contrastive_loss.temporal_loss
        self.temporal_soft_loss = self.Contrastive_loss.temporal_soft_loss
        self.accuracy = accuracy
                    
        
    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.mixed_precision)
        
        save_config_file(self.args.log_dir, self.args) 
        
        n_iter = 0
        best_loss = float('inf')
        logging.info(f"Start SimCLRTemporalSpatial training for {self.args.epochs} epochs.")
        logging.info(f"Training on device: {self.args.device}.")
        
        for epoch in range(self.args.epochs):
            # permute, transpose后需要用contiguous() 才能进行reshape
            self.model.train()
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}"):
                anchors = batch["anchors"].to(self.args.device) # [batch_size, n_views, C, H, W]
                # anchors = anchors.transpose(0, 1).contiguous()   # [n_views, batch_size, C, H, W]
                # anchors = anchors.view(-1, anchors.size(2), 
                #                                 anchors.size(3), anchors.size(4)).to(self.args.device)  
                # [2*batch_size, C, H, W]
                
                temporal_negatives = batch["temporal_negatives"].to(self.args.device)  # [batch_size, num_neg, C, H, W]
                
                with autocast(enabled=self.args.mixed_precision):
                    features = self.model(anchors.view(-1, anchors.size(2), 
                                                            anchors.size(3), 
                                                            anchors.size(4)))  # [batch_size*n_views, feature_dim=128]
                    features = features.view(self.args.batch_size, self.args.n_views, -1) 
                    # features: [batch_size, n_views, feature_dim=128]
                    
                    
                    temporal_negatives = self.model(temporal_negatives.view(-1, 
                                                                            temporal_negatives.size(2), 
                                                                            temporal_negatives.size(3), 
                                                                            temporal_negatives.size(4)))
                    temporal_negatives = temporal_negatives.view(self.args.batch_size, self.args.num_neg, -1)
                    # temporal_negatives: [batch_size, num_neg, feature_dim=128]
                    
                    # 计算4个损失函数
                    # 1. spatial_group_smoothing_loss
                    spatial_loss = self.spatial_loss(features)
                    
                    # 2. temporal_loss  
                    temporal_logits, temporal_labels_ce, temporal_loss = self.temporal_loss(features=features, temporal_negatives=temporal_negatives)
                    
                    # 3. temporal_soft_loss (需要虚拟参数)
                    anchors = features[:, 0, :]  # [batch_size, feature_dim]
                    actual_batch_size = anchors.size(0)
                    
                    dummy_negative_ratios = torch.ones(actual_batch_size, self.args.num_neg, 12).to(self.args.device)
                    dummy_label_emb = torch.randn(12, 768).to(self.args.device)
                    temporal_soft_logits, temporal_soft_labels, temporal_soft_loss, indices, _ = self.temporal_soft_loss(
                        anchors, temporal_negatives, 
                        dummy_negative_ratios, 
                        dummy_label_emb
                    )
                    
                    # 4. group_spatial_temporal_loss
                    spatial_temporal_loss = self.spatial_temporal_loss(
                        anchors, temporal_negatives, indices
                    )
            
                    loss = (self.args.spatial_weight * spatial_loss + 
                           self.args.temporal_weight * temporal_loss + 
                           self.args.temporal_soft_weight * temporal_soft_loss +
                           self.args.spatial_temporal_weight * spatial_temporal_loss)   
                    

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                    
                # 日志记录
                if n_iter % self.args.log_every_n_steps == 0:
                    metrics = {
                        'loss/spatial': spatial_loss.item(),
                        'loss/temporal': temporal_loss.item(),
                        'loss/temporal_soft': temporal_soft_loss.item(),
                        'loss/spatial_temporal': spatial_temporal_loss.item(),
                        'loss/total': loss.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch
                    }
                    
                    # 计算精度(只有temporal_loss有logits)
                    if temporal_logits is not None and temporal_labels_ce is not None:
                        top1_temporal_acc, top5_temporal_acc = self.accuracy(temporal_logits, temporal_labels_ce, topk=(1, 5))
                        metrics.update({
                            'accuracy/top1_temporal': top1_temporal_acc[0].item() if hasattr(top1_temporal_acc, 'item') else top1_temporal_acc,
                            'accuracy/top5_temporal': top5_temporal_acc[0].item() if hasattr(top5_temporal_acc, 'item') else top5_temporal_acc
                        })
                    
                    wandb.log(metrics, step=n_iter)

                n_iter += 1
                
            # 调整学习率
            if epoch >= self.args.warmup_epochs:
                self.scheduler.step()

            # 记录每个 epoch 的损失
            epoch_metrics = {
                'epoch/spatial_loss': spatial_loss.item(),
                'epoch/temporal_loss': temporal_loss.item(),
                'epoch/temporal_soft_loss': temporal_soft_loss.item(),
                'epoch/spatial_temporal_loss': spatial_temporal_loss.item(),
                'epoch/total_loss': loss.item(),
                'epoch/learning_rate': self.optimizer.param_groups[0]['lr']
            }
            
            if temporal_logits is not None and temporal_labels_ce is not None:
                top1_temporal_acc, top5_temporal_acc = self.accuracy(temporal_logits, temporal_labels_ce, topk=(1, 5))
                epoch_metrics.update({
                    'epoch/top1_temporal': top1_temporal_acc[0].item() if hasattr(top1_temporal_acc, 'item') else top1_temporal_acc,
                    'epoch/top5_temporal': top5_temporal_acc[0].item() if hasattr(top5_temporal_acc, 'item') else top5_temporal_acc
                })
            
            wandb.log(epoch_metrics, step=epoch)
            
            logging.debug(f"Epoch: {epoch+1}\tSpatial Loss: {spatial_loss.item():.4f}\tTemporal Loss: {temporal_loss.item():.4f}\tSpatial Temporal Loss: {spatial_temporal_loss.item():.4f}\tTotal Loss: {loss.item():.4f}\tLearning Rate: {self.optimizer.param_groups[0]['lr']:.4f}")

            # 检查点保存(每50轮)
            if (epoch + 1) % 50 == 0:
                checkpoint_name = f'checkpoint_epoch_{epoch+1:04d}.pth'
                checkpoint_path = os.path.join(self.args.log_dir, checkpoint_name)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'loss': loss.item(),
                    'wandb_run_id': wandb.run.id
                }, is_best=False, filename=checkpoint_path)
                logging.info(f"Checkpoint saved at {checkpoint_path}")
            
            # 保存最佳模型
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_checkpoint_path = os.path.join(self.args.log_dir, 'best.pth')
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'best_loss': best_loss,
                    'wandb_run_id': wandb.run.id
                }, is_best=True, filename=best_checkpoint_path)
                logging.info(f"New best model saved with loss: {best_loss:.4f}")
        
        logging.info("Training has finished.")
        # 保存最终模型
        final_checkpoint_name = f'final_checkpoint_epoch_{self.args.epochs:04d}.pth'
        final_checkpoint_path = os.path.join(self.args.log_dir, final_checkpoint_name)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'final_loss': loss.item(),
            'wandb_run_id': wandb.run.id
        }, is_best=False, filename=final_checkpoint_path)
        logging.info(f"Final model checkpoint saved at {final_checkpoint_path}")
        
        wandb.finish()
