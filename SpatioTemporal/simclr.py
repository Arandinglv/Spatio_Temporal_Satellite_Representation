import logging
import torch
import torch.nn.functional as F
from datetime import datetime
import os
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
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
        self.writer = SummaryWriter(log_dir=self.args.log_dir)
        logging.basicConfig(
            filename=os.path.join(self.writer.log_dir, 'training.log'),
            level=logging.DEBUG,
            format='%(asctime)s %(levelname)s:%(message)s'
        )
        self.Contrastive_loss = ContrastiveLoss(self.args)
        self.spatial_loss = self.Contrastive_loss.spatial_loss
        self.spatial_temporal_loss = self.Contrastive_loss.spatial_temporal_loss
        self.temporal_loss = self.Contrastive_loss.temporal_loss
        self.accuracy = accuracy
                    
        
    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.mixed_precision)
        
        save_config_file(self.writer.log_dir, self.args) 
        
        n_iter = 0
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
                    
                    # 计算损失
                    spatial_logits, spatial_labels_ce, spatial_loss = self.spatial_loss(features)
                    temporal_logits, \
                        temporal_labels_ce, \
                            temporal_loss = self.temporal_loss(features=features, temporal_negatives=temporal_negatives)
                    
                    spatial_temporal_logits, \
                        spatial_temporal_labels, \
                            spatial_temporal_loss = self.spatial_temporal_loss(features=features, temporal_negatives=temporal_negatives)
            
                    loss = 0.4 * spatial_loss + 0.4 * temporal_loss + 0.2 * spatial_temporal_loss   
                    

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                    
                # 日志记录
                if n_iter % self.args.log_every_n_steps == 0:
                    top1_spatial_acc, top5_spatial_acc = self.accuracy(spatial_logits, spatial_labels_ce, topk=(1, 5))
                    top1_temporal_acc, top5_temporal_acc = self.accuracy(temporal_logits, temporal_labels_ce, topk=(1, 5))
                    top1_spatial_temporal_acc, top5_spatial_temporal_acc = \
                        self.accuracy(spatial_temporal_logits, spatial_temporal_labels, topk=(1, 5))
                    # loss 计算
                    self.writer.add_scalar('loss/spatial', spatial_loss.item(), global_step=n_iter)
                    self.writer.add_scalar('loss/temporal', temporal_loss.item(), global_step=n_iter)
                    self.writer.add_scalar('loss/spatial_temporal', spatial_temporal_loss.item(), global_step=n_iter)
                    self.writer.add_scalar('loss/total', loss.item(), global_step=n_iter)
                    self.writer.add_scalar('accuracy/top1_spatial', top1_spatial_acc, global_step=n_iter)
                    self.writer.add_scalar('accuracy/top5_spatial', top5_spatial_acc, global_step=n_iter)
                    self.writer.add_scalar('accuracy/top1_temporal', top1_temporal_acc, global_step=n_iter)
                    self.writer.add_scalar('accuracy/top5_temporal', top5_temporal_acc, global_step=n_iter)
                    self.writer.add_scalar('accuracy/top1_spatial_temporal', top1_spatial_temporal_acc, global_step=n_iter)
                    self.writer.add_scalar('accuracy/top5_spatial_temporal', top5_spatial_temporal_acc, global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], global_step=n_iter)

                n_iter += 1
                
            # 调整学习率
            if epoch >= self.args.warmup_epochs:
                self.scheduler.step()

            # 记录每个 epoch 的损失
            logging.debug(f"Epoch: {epoch+1}\
                            \tSpatial Loss: {spatial_loss.item():.4f}\
                            \tTemporal Loss: {temporal_loss.item():.4f}\
                            \tSpatial Temporal Loss: {spatial_temporal_loss.item():.4f}\
                            \tTotal Loss: {loss.item():.4f}\
                                \tSpatial Top-1 Accuracy: {top1_spatial_acc[0]:.4f}\
                                \tSpatial Top-5 Accuracy: {top5_spatial_acc[0]:.4f}\
                                \tTemporal Top-1 Accuracy: {top1_temporal_acc[0]:.4f}\
                                \tTemporal Top-5 Accuracy: {top5_temporal_acc[0]:.4f}\
                                \tSpatial Temporal Top-1 Accuracy: {top1_spatial_temporal_acc[0]:.4f}\
                                \tSpatial Temporal Top-5 Accuracy: {top5_spatial_temporal_acc[0]:.4f}\
                                    \tLearning Rate: {self.optimizer.param_groups[0]['lr']:.4f}")
            # logging.debug(f"Epoch: {epoch+1}\tSpatial Loss: {spatial_loss.item():.4f}\
            #                 \tSpatial Temporal Loss: {spatial_temporal_loss.item():.4f}\
            #                   \tTotal Loss: {loss.item():.4f}\
            #                       \tLearning Rate: {self.optimizer.param_groups[0]['lr']:.4f}\
            #                           \tSpatial Top-1 Accuracy: {top1_spatial_acc[0]:.4f}\
            #                               \tSpatial Top-5 Accuracy: {top5_spatial_acc[0]:.4f}")
            self.writer.add_scalar('epoch/spatial_loss', spatial_loss.item(), global_step=epoch)
            self.writer.add_scalar('epoch/temporal_loss', temporal_loss.item(), global_step=epoch)
            self.writer.add_scalar('epoch/spatial_temporal_loss', spatial_temporal_loss.item(), global_step=epoch)
            self.writer.add_scalar('epoch/total_loss', loss.item(), global_step=epoch)
            self.writer.add_scalar('epoch/learning_rate', self.optimizer.param_groups[0]['lr'], global_step=epoch)
            self.writer.add_scalar('epoch/top1_spatial', top1_spatial_acc, global_step=epoch)
            self.writer.add_scalar('epoch/top5_spatial', top5_spatial_acc, global_step=epoch)
            self.writer.add_scalar('epoch/top1_temporal', top1_temporal_acc, global_step=epoch)
            self.writer.add_scalar('epoch/top5_temporal', top5_temporal_acc, global_step=epoch)
            self.writer.add_scalar('epoch/top1_spatial_temporal', top1_spatial_temporal_acc, global_step=epoch)
            self.writer.add_scalar('epoch/top5_spatial_temporal', top5_spatial_temporal_acc, global_step=epoch)
            
            # break  

        logging.info("Training has finished.")
        # 保存模型检查点
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
