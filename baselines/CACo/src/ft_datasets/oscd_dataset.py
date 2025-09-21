import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import numpy as np
from itertools import product


class OSCDRGBDataset(Dataset):

    def __init__(self, split='train', transform=None, patch_size=96):
        self.split = split
        self.transform = transform
        self.patch_size = patch_size

        # Load OSCD RGB dataset from HuggingFace
        dataset = load_dataset("blanchon/OSCD_RGB", split=split)

        # Generate patches following SECO approach
        self.samples = []
        for item in dataset:
            img1 = item['image1'].convert('RGB')
            img2 = item['image2'].convert('RGB')
            mask = item['mask'].convert('L')

            w, h = img1.size

            # Generate patch limits using itertools.product like SECO
            limits = product(range(0, w, patch_size), range(0, h, patch_size))
            for l in limits:
                x, y = l
                if x + patch_size <= w and y + patch_size <= h:
                    self.samples.append({
                        'img1': img1,
                        'img2': img2,
                        'mask': mask,
                        'limits': (x, y, x + patch_size, y + patch_size)
                    })

    def __getitem__(self, index):
        sample = self.samples[index]

        # Crop patches using limits like SECO
        img1 = sample['img1'].crop(sample['limits'])
        img2 = sample['img2'].crop(sample['limits'])
        mask = sample['mask'].crop(sample['limits'])

        if self.transform is not None:
            img1, img2, mask = self.transform(img1, img2, mask)

        return img1, img2, mask

    def __len__(self):
        return len(self.samples)
