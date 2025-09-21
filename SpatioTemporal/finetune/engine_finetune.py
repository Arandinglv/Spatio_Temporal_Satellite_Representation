import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from sklearn.metrics import r2_score
from timm.utils import accuracy


def train_one_epoch(model, head, criterion, data_loader, optimizer, device, epoch, args, model_freeze=True):
    if model_freeze:
        model.eval()  # Use eval mode for frozen model to disable dropout/batchnorm updates
    else:
        model.train()
    if head is not None:
        head.train(True)
    
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(data_loader):
        # Unpack batch based on dataset type
        if len(batch) == 2:
            samples, targets = batch
        else:
            samples, targets = batch[0], batch[-1]
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        
        # Forward pass
        if model_freeze:
            with torch.no_grad():
                features = model(samples)
        else:
            features = model(samples)
        
        if head is not None:
            outputs = head(features)
        else:
            outputs = features
        
        loss = criterion(outputs, targets)
        
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            raise ValueError(f"Loss is {loss.item()}, stopping training")
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Disable frequent wandb logging for faster training
        # if batch_idx % 100 == 0 and hasattr(args, 'wandb') and args.wandb:
        #     wandb.log({
        #         'train_loss_step': loss.item(),
        #         'epoch': epoch,
        #         'batch': batch_idx
        #     })
    
    avg_loss = total_loss / num_batches
    print(f"Train Epoch {epoch}: Average Loss = {avg_loss:.4f}")
    
    return {"loss": avg_loss}


@torch.no_grad()
def evaluate(data_loader, model, head, device, criterion, args):
    """Unified evaluation function for all dataset types"""
    model.eval()
    if head is not None:
        head.eval()
    
    # Determine task type from first batch
    task_type = None
    for samples, targets in data_loader:
        if isinstance(targets, torch.FloatTensor) or targets.dtype == torch.float32:
            task_type = "regression"
        else:
            task_type = "classification"
        break
    
    total_loss = 0
    num_batches = 0
    all_targets = []
    all_predictions = []
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    
    for batch in data_loader:
        if len(batch) == 2:
            samples, targets = batch
        else:
            samples, targets = batch[0], batch[-1]
            
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass
        features = model(samples)
        if head is not None:
            outputs = head(features)
        else:
            outputs = features
            
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        num_batches += 1
        
        batch_size = samples.shape[0]
        total_samples += batch_size
        
        if task_type == "regression":
            # Collect for regression metrics
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
        else:
            # Classification accuracy
            if args.dataset_type == 'eurosat':
                # Multi-class classification
                acc1, acc5 = accuracy(outputs, targets, topk=(1, min(5, outputs.size(1))))
                correct_top1 += acc1.item() * batch_size / 100
                correct_top5 += acc5.item() * batch_size / 100
            elif args.dataset_type == 'bigearthnet':
                # Multi-label classification
                pred = (torch.sigmoid(outputs) > 0.5).float()
                correct_top1 += ((pred == targets).sum(dim=1) == targets.size(1)).float().sum().item()
                correct_top5 = correct_top1  # Not applicable for multi-label
    
    avg_loss = total_loss / num_batches
    
    if task_type == "regression":
        # Compute regression metrics
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        if all_targets.ndim > 1:
            all_targets = all_targets.flatten()
        if all_predictions.ndim > 1:
            all_predictions = all_predictions.flatten()
        
        mse = np.mean((all_predictions - all_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_predictions - all_targets))
        
        # MAPE
        mask = all_targets != 0
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((all_targets[mask] - all_predictions[mask]) / all_targets[mask])) * 100
        else:
            mape = float('inf')
        
        # R²
        r2 = r2_score(all_targets, all_predictions)
        
        print(f'* MSE {mse:.4f} RMSE {rmse:.4f} MAE {mae:.4f} MAPE {mape:.2f}% R² {r2:.4f} loss {avg_loss:.4f}')
        
        return {
            'loss': avg_loss,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'acc1': 0,  # dummy for compatibility
            'acc5': 0   # dummy for compatibility
        }
    else:
        # Classification metrics
        acc1 = correct_top1 / total_samples
        acc5 = correct_top5 / total_samples
        
        if args.dataset_type == 'eurosat':
            print(f'* Acc@1 {acc1*100:.2f}% Acc@5 {acc5*100:.2f}% loss {avg_loss:.4f}')
        elif args.dataset_type == 'bigearthnet':
            print(f'* Sample Acc {acc1*100:.2f}% loss {avg_loss:.4f}')
        
        return {
            'loss': avg_loss,
            'acc1': acc1,
            'acc5': acc5
        }


# Alias for backward compatibility
def evaluate_indicator(data_loader, model, head, device, criterion, args):
    """Alias for regression evaluation"""
    return evaluate(data_loader, model, head, device, criterion, args)