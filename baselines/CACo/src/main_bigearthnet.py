from pathlib import Path
from argparse import ArgumentParser

import torch
from torch.nn import BCEWithLogitsLoss
from torchvision.models import resnet50
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Precision, Recall, F1Score, AUROC
import wandb

from ft_datasets.bigearthnet_hf_datamodule import BigearthnetHFDataModule


class BigearthnetClassifier(LightningModule):

    def __init__(self, backbone, num_classes=19, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])

        self.backbone = backbone
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Get backbone output features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_features = self.backbone(dummy_input).shape[1]

        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(backbone_features, num_classes)
        )

        # Loss function for multi-label classification
        self.criterion = BCEWithLogitsLoss()

        # Metrics for multi-label classification
        self.train_precision = Precision(task='multilabel', num_labels=num_classes, average='macro')
        self.train_recall = Recall(task='multilabel', num_labels=num_classes, average='macro')
        self.train_f1 = F1Score(task='multilabel', num_labels=num_classes, average='macro')
        self.train_auroc = AUROC(task='multilabel', num_labels=num_classes, average='macro')

        self.val_precision = Precision(task='multilabel', num_labels=num_classes, average='macro')
        self.val_recall = Recall(task='multilabel', num_labels=num_classes, average='macro')
        self.val_f1 = F1Score(task='multilabel', num_labels=num_classes, average='macro')
        self.val_auroc = AUROC(task='multilabel', num_labels=num_classes, average='macro')

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)

        # Convert logits to probabilities for metrics
        probs = torch.sigmoid(logits)

        # Update metrics
        self.train_precision(probs, targets.int())
        self.train_recall(probs, targets.int())
        self.train_f1(probs, targets.int())
        self.train_auroc(probs, targets.int())

        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/precision', self.train_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/auroc', self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)

        # Convert logits to probabilities for metrics
        probs = torch.sigmoid(logits)

        # Update metrics
        self.val_precision(probs, targets.int())
        self.val_recall(probs, targets.int())
        self.val_f1(probs, targets.int())
        self.val_auroc(probs, targets.int())

        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # Only train the classification head, freeze backbone
        params = self.classifier.parameters()
        optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--backbone_type', type=str, default='caco')
    parser.add_argument('--ckpt_path', type=str,
                       default='/hpc2hdd/home/qwang650/project/yutianjiang/stmodel/weights/caco/resnet50_caco_geo_1m_200.pth')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--train_fraction', type=float, default=0.1, help='Fraction of training data to use (default: 0.1 for 10%)')
    args = parser.parse_args()

    # Initialize datamodule
    datamodule = BigearthnetHFDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        train_fraction=args.train_fraction
    )

    # Initialize backbone
    if args.backbone_type == 'random':
        backbone = resnet50(pretrained=False)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    elif args.backbone_type == 'imagenet':
        backbone = resnet50(pretrained=True)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    elif args.backbone_type == 'caco':
        backbone = resnet50(pretrained=False)
        state_dict = torch.load(args.ckpt_path, map_location='cpu')
        backbone.load_state_dict(state_dict, strict=False)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    else:
        raise ValueError(f"Unknown backbone type: {args.backbone_type}")

    # Freeze backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False

    # Initialize model
    model = BigearthnetClassifier(
        backbone=backbone,
        num_classes=datamodule.num_classes,
        learning_rate=args.learning_rate
    )

    # Setup experiment name
    experiment_name = f"caco_bigearthnet_{args.backbone_type}_lr{args.learning_rate}_bs{args.batch_size}"

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project="linprobe",
        name="caco",
        save_dir=str(Path.cwd() / 'logs' / 'bigearthnet'),
        log_model=True
    )

    # Add hyperparameters to wandb
    wandb_logger.experiment.config.update({
        'backbone_type': args.backbone_type,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'learning_rate': args.learning_rate,
        'max_epochs': args.max_epochs,
        'train_fraction': args.train_fraction,
        'num_classes': datamodule.num_classes
    })

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{val/f1:.3f}',
        monitor='val/f1',
        mode='max',
        save_top_k=3,
        save_last=True
    )

    # Initialize trainer
    trainer = Trainer(
        devices=args.gpus,
        accelerator='gpu',
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=args.max_epochs,
        precision='16-mixed',  # Updated for current PyTorch Lightning
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Train model
    trainer.fit(model, datamodule=datamodule)

    # Close wandb run
    wandb.finish()
