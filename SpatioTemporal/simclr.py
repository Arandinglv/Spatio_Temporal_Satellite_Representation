import logging
import torch
import torch.nn.functional as F
from datetime import datetime
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
        
        # Set wandb API key and login
        os.environ["WANDB_API_KEY"] = "8f2fc4596fca723c90b10053eeaead122e46603a"
        wandb.login()
        
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

        for epoch in range(self.args.epochs):
            # !!!!!!!脚本传入的batch_size其实就是group_num
            # 只不过真正的batch_size = group_num * group_element
            train_loader.dataset.current_epoch = epoch
            self.model.train()
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}"):
                self.optimizer.zero_grad()
                # [batch_size, group_element, n_views, C, H, W]
                anchors = batch["anchors"].to(self.args.device)
                temporal_negatives = batch["temporal_negatives"].to(self.args.device)
                temporal_negatives_ratios = batch["temporal_negatives_ratios"].to(self.args.device)

                batch_size, group_element, n_views, C, H, W = anchors.shape
                _, _, num_neg, _, _, _ = temporal_negatives.shape

                # loss = 0

                with autocast(enabled=self.args.mixed_precision):
                    anchors = anchors.view(-1, C, H, W)  # [batch_size*group_element*n_views, C, H, W]
                    features = self.model(anchors)  # [batch_size*group_element*n_views,
                    features_dim = features.shape[-1]
                    features = features.view(batch_size * group_element, n_views, features_dim)

                    temporal_negatives = temporal_negatives.view(-1, C, H, W)
                    temporal_negatives = self.model(temporal_negatives).view(
                        batch_size * group_element, num_neg, features_dim
                    )

                    negative_ratios = temporal_negatives_ratios.view(batch_size * group_element, num_neg, -1)

                    # spatial loss
                    spatial_loss = self.spatial_group_smoothing_loss(features)

                    # temporal hard loss
                    temporal_logits, temporal_labels_ce, temporal_loss = self.temporal_loss(
                        features=features,
                        temporal_negatives=temporal_negatives
                    )
                    # temporal soft loss
                    anchors_single = features[:, 0, :]  # [batch_size*group_element, feature_dim]
                    soft_logits, soft_labels, soft_loss, indices, soft_matrix = self.temporal_soft_loss(
                        anchors=anchors_single,
                        temporal_negatives=temporal_negatives,
                        negative_ratios=negative_ratios,
                        label_emb=self.label_emb.to(self.args.device)
                    )
                    group_st_loss = self.group_spatial_temporal_loss(
                        anchors=anchors_single,
                        temporal_negatives=temporal_negatives,
                        indices=indices
                    )

                loss = 0.3 * spatial_loss + 0.3 * temporal_loss + 0.2 * soft_loss 
                loss_2 = + 0.2 * group_st_loss

                scaler.scale(loss).backward(retain_graph=True)
                scaler.scale(loss_2).backward()
                scaler.step(self.optimizer)
                scaler.update()
                if n_iter % self.args.log_every_n_steps == 0:
                    metrics = {
                        'loss/temporal': temporal_loss.item(),
                        'loss/spatial': spatial_loss.item(),
                        'loss/temporal_soft': soft_loss.item(),
                        'loss/group_spatial_temporal': group_st_loss.item(),
                        'loss/total': loss.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'step': n_iter,
                        'epoch': epoch
                    }

                    # 计算准确率（只有temporal_loss有logits）
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

            # 记录每个epoch的指标到wandb
            epoch_metrics = {
                'epoch/temporal_loss': temporal_loss.item(),
                'epoch/spatial_loss': spatial_loss.item(),
                'epoch/temporal_soft_loss': soft_loss.item(),
                'epoch/group_spatial_temporal_loss': group_st_loss.item(),
                'epoch/total_loss': loss.item(),
                'epoch/learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_num': n_iter + 1
            }

            if temporal_logits is not None and temporal_labels_ce is not None:
                top1_temporal_acc, top5_temporal_acc = self.accuracy(temporal_logits, temporal_labels_ce, topk=(1, 5))
                epoch_metrics.update({
                    'epoch/top1_temporal': top1_temporal_acc[0].item() if hasattr(top1_temporal_acc, 'item') else top1_temporal_acc,
                    'epoch/top5_temporal': top5_temporal_acc[0].item() if hasattr(top5_temporal_acc, 'item') else top5_temporal_acc
                })

            wandb.log(epoch_metrics, step=epoch)
            