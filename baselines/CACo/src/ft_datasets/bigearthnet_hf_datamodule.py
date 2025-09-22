from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from PIL import Image
import torch
import numpy as np

from ft_datasets.bigearthnet_hf_dataset import BigearthnetHFDataset


class BigearthnetHFDataModule(LightningDataModule):

    def __init__(self, batch_size=32, num_workers=4, image_size=224, train_fraction=0.1, seed=42):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_fraction = train_fraction
        self.seed = seed

    @property
    def num_classes(self):
        return 19

    def setup(self, stage=None):
        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load full training dataset
        full_train_dataset = BigearthnetHFDataset(
            split='train',
            transform=train_transform
        )

        # Create subset of training data (default 10%)
        if self.train_fraction < 1.0:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            train_size = len(full_train_dataset)
            subset_size = int(train_size * self.train_fraction)

            # Create random indices for subset - convert to list to avoid tensor indexing issues
            indices = torch.randperm(train_size)[:subset_size].tolist()
            self.train_dataset = Subset(full_train_dataset, indices)

            print(f"Using {subset_size}/{train_size} training samples ({self.train_fraction*100:.1f}%)")
        else:
            self.train_dataset = full_train_dataset

        # Use test split as validation since validation might not exist
        self.val_dataset = BigearthnetHFDataset(
            split='test',
            transform=val_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )