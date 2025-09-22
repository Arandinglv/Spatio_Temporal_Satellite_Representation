import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import numpy as np


# BigEarthNet-S2 labels (19 classes)
BIGEARTHNET_LABELS = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters'
]


class BigearthnetHFDataset(Dataset):

    def __init__(self, split='train', transform=None):
        self.split = split
        self.transform = transform
        self.num_classes = len(BIGEARTHNET_LABELS)

        # Load BigEarthNet dataset from HuggingFace
        self.dataset = load_dataset("danielz01/BigEarthNet-S2-v1.0", split=split)

    def __getitem__(self, index):
        try:
            # Ensure index is an integer
            if isinstance(index, torch.Tensor):
                index = index.item()

            sample = self.dataset[index]

            # Get RGB image
            img = sample['img']
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            img = img.convert('RGB')

            # Get labels and convert to multi-hot encoding
            labels = sample['labels']
            target = self._labels_to_multihot(labels)

            if self.transform is not None:
                img = self.transform(img)

            return img, target
        except Exception as e:
            print(f"Error loading sample {index}: {e}")
            print(f"Index type: {type(index)}")
            raise

    def _labels_to_multihot(self, labels):
        """Convert labels to multi-hot encoding"""
        target = torch.zeros(self.num_classes, dtype=torch.float32)

        if isinstance(labels, str):
            labels = [labels]
        elif not isinstance(labels, (list, tuple)):
            labels = list(labels)

        for label in labels:
            if isinstance(label, str) and label in BIGEARTHNET_LABELS:
                idx = BIGEARTHNET_LABELS.index(label)
                target[idx] = 1.0
            elif isinstance(label, int) and 0 <= label < self.num_classes:
                target[label] = 1.0

        return target

    def __len__(self):
        return len(self.dataset)