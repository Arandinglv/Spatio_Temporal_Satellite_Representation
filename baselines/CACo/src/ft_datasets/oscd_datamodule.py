import random

from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from pytorch_lightning import LightningDataModule

from ft_datasets.oscd_dataset import OSCDRGBDataset


class RandomFlip:

    def __call__(self, *xs):
        if random.random() > 0.5:
            xs = tuple(TF.hflip(x) for x in xs)
        return xs


class RandomRotation:

    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, *xs):
        angle = random.choice(self.angles)
        return tuple(TF.rotate(x, angle) for x in xs)


class RandomSwap:

    def __call__(self, x1, x2, y):
        if random.random() > 0.5:
            return x2, x1, y
        else:
            return x1, x2, y


class ToTensor:

    def __call__(self, *xs):
        return tuple(TF.to_tensor(x) for x in xs)


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *xs):
        for t in self.transforms:
            xs = t(*xs)
        return xs


class ChangeDetectionDataModule(LightningDataModule):

    def __init__(self, patch_size=96):
        super().__init__()
        self.patch_size = patch_size

    def setup(self, stage=None):
        self.train_dataset = OSCDRGBDataset(
            split='train',
            transform=Compose([RandomFlip(), RandomRotation(), RandomSwap(), ToTensor()]),
            patch_size=self.patch_size
        )
        self.val_dataset = OSCDRGBDataset(
            split='test',
            transform=ToTensor(),
            patch_size=self.patch_size
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            drop_last=False,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True
        )
