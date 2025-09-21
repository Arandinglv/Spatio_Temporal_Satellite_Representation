from pathlib import Path
from argparse import ArgumentParser

import torch
from torch.nn import BCEWithLogitsLoss
from torchvision.models import resnet50
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Precision, Recall, F1Score

from ft_datasets.oscd_datamodule import ChangeDetectionDataModule
from models.segmentation import get_segmentation_model


class SiamSegment(LightningModule):

    def __init__(self, backbone, feature_indices, feature_channels):
        super().__init__()
        self.model = get_segmentation_model(backbone, feature_indices, feature_channels)
        # Use weighted BCE for class imbalance
        self.criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(10.0))
        self.prec = Precision(task='binary', num_classes=2, threshold=0.5)
        self.rec = Recall(task='binary', num_classes=2, threshold=0.5)
        self.f1 = F1Score(task='binary', num_classes=2, threshold=0.5)

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def training_step(self, batch, batch_idx):
        img_1, img_2, mask, pred, loss, prec, rec, f1 = self.shared_step(batch)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/precision', prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/recall', rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img_1, img_2, mask, pred, loss, prec, rec, f1 = self.shared_step(batch)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/precision', prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/recall', rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def shared_step(self, batch):
        img_1, img_2, mask = batch

        # Convert mask to binary: any non-zero value becomes 1
        mask_binary = (mask > 0).float()

        out = self(img_1, img_2)
        pred = torch.sigmoid(out)
        loss = self.criterion(out, mask_binary)

        pred_binary = (pred > 0.5).int()
        mask_binary_int = mask_binary.int()

        prec = self.prec(pred_binary, mask_binary_int)
        rec = self.rec(pred_binary, mask_binary_int)
        f1 = self.f1(pred_binary, mask_binary_int)
        return img_1, img_2, mask, pred, loss, prec, rec, f1

    def configure_optimizers(self):
        # params = self.model.parameters()
        params = set(self.model.parameters()).difference(self.model.encoder.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--backbone_type', type=str, default='caco')
    parser.add_argument('--ckpt_path', type=str,
                       default='/hpc2hdd/home/qwang650/project/yutianjiang/stmodel/weights/caco/resnet50_caco_geo_1m_200.pth')
    parser.add_argument('--max_epochs', type=int, default=100)
    args = parser.parse_args()

    datamodule = ChangeDetectionDataModule(patch_size=args.patch_size)

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

    # ResNet50 feature channels
    model = SiamSegment(
        backbone,
        feature_indices=(0, 4, 5, 6, 7),
        feature_channels=(64, 256, 512, 1024, 2048)
    )

    experiment_name = f"{args.backbone_type}_patch{args.patch_size}"
    logger = TensorBoardLogger(save_dir=str(Path.cwd() / 'logs' / 'oscd'), name=experiment_name)

    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{val/f1:.3f}',
        monitor='val/f1',
        mode='max',
        save_top_k=3
    )

    trainer = Trainer(
        devices=args.gpus,
        accelerator='gpu',
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=args.max_epochs,
        precision=16
    )

    trainer.fit(model, datamodule=datamodule)
