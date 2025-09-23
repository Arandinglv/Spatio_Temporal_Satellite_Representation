import logging
import torch
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
        self.train_sampler = kwargs.get('train_sampler', None)

        # 使用DDP包装模型 - SOTA级配置
        if hasattr(self.args, 'distributed') and self.args.distributed:
            # 同步BatchNorm避免原地操作冲突
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(
                self.model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,  # 支持多阶段backward
                broadcast_buffers=False,  # 避免buffer冲突
                gradient_as_bucket_view=True  # SOTA级梯度优化
            )
            if self.args.rank == 0:
                print(f"Model wrapped with DDP + SyncBatchNorm on rank {self.args.rank}")
        
        # 1. 自动创建日志和检查点目录 (仅在rank 0进行)
        # 日志和检查点都将保存在 self.args.log_dir 指定的路径下
        self.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
        if not hasattr(self.args, 'rank') or self.args.rank == 0:
            os.makedirs(self.args.log_dir, exist_ok=True)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 2. 初始化 wandb (仅在rank 0进行)
        # Set wandb API key and login
        self.use_wandb = not hasattr(self.args, 'rank') or self.args.rank == 0
        if self.use_wandb:
            os.environ["WANDB_API_KEY"] = "8f2fc4596fca723c90b10053eeaead122e46603a"
            wandb.login()

            wandb.init(
                project=self.args.wandb_project,
                name=self.args.experiment_name,
                config=vars(self.args),
                mode="online"
            )
        
        # 3. 初始化日志记录器 (logging) (仅在rank 0进行)
        if not hasattr(self.args, 'rank') or self.args.rank == 0:
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

        # 异常检测已完成，DDP问题已修复
        # torch.autograd.set_detect_anomaly(True)  # 已关闭以提升性能

        # 只在rank 0保存配置文件和打印日志
        if not hasattr(self.args, 'rank') or self.args.rank == 0:
            save_config_file(self.args.log_dir, self.args)
            logging.info(f"Start SimCLRTemporalSpatial training for {self.args.epochs} epochs.")
            logging.info(f"Training on device: {self.args.device}.")
            logging.info(f"Logs and checkpoints will be saved in: {self.args.log_dir}")

        # DDP同步所有进程
        if hasattr(self.args, 'distributed') and self.args.distributed:
            dist.barrier()

        n_iter = 0
        label_emb = self.label_emb.to(self.args.device)

        for epoch in range(self.args.epochs):
            # 设置epoch用于DDP的正确数据shuffle
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            train_loader.dataset.current_epoch = epoch
            self.model.train()
            
            # 只在rank 0显示进度条
            iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}") if (not hasattr(self.args, 'rank') or self.args.rank == 0) else train_loader
            for batch in iterator:
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

                    # Combine losses - SOTA分阶段反向传播设计
                    loss = 0.9 * spatial_loss + 0.2 * temporal_loss + 0.3 * group_st_loss
                    loss_2 = 0.2 * soft_loss
                    total_loss = loss + loss_2

                # Backward pass and optimization - ICML SOTA级双阶段设计
                # 第一阶段：保留计算图用于软损失的反向传播
                scaler.scale(loss_2).backward(retain_graph=True)
                # 第二阶段：完整反向传播主损失
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

                    if self.use_wandb:
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

            if self.use_wandb:
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

            if not hasattr(self.args, 'rank') or self.args.rank == 0:
                log_message = (f"Epoch {epoch+1}/{self.args.epochs} Summary: "
                               f"Total Loss={total_loss.item():.4f}, "
                               f"Top1 Acc={top1_acc:.4f}, "
                               f"LR={self.optimizer.param_groups[0]['lr']:.6f}")
                logging.info(log_message)
            
            # Save checkpoint every epoch (仅在rank 0进行)
            if ((epoch + 1) % 1 == 0 or (epoch + 1) == self.args.epochs) and (not hasattr(self.args, 'rank') or self.args.rank == 0):
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1:04d}.pth')
                # 获取实际模型状态字典 (去除DDP包装)
                model_state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.args.arch,
                    'state_dict': model_state_dict,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                }, is_best=False, filename=checkpoint_path)

                logging.info(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
                # wandb.save(checkpoint_path) # Optional: save checkpoint to wandb cloud

        # --- End of Training ---

        if not hasattr(self.args, 'rank') or self.args.rank == 0:
            logging.info("Training has finished.")
            # Save final model
            final_checkpoint_path = os.path.join(self.checkpoint_dir, 'final_model.pth')
            # 获取实际模型状态字典 (去除DDP包装)
            model_state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
            save_checkpoint({
                'epoch': self.args.epochs,
                'arch': self.args.arch,
                'state_dict': model_state_dict,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            }, is_best=False, filename=final_checkpoint_path)

            logging.info(f"Final model saved to {final_checkpoint_path}")
            # wandb.save(final_checkpoint_path) # Optional: save final model to wandb cloud

        if self.use_wandb:
            wandb.finish()