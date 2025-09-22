import numpy as np
import pandas as pd
from ft_datasets.Indicator import IndicatorDataset
import torchvision.transforms as transforms

def create_indicator_datasets(args):
    """To create train, val, test datasets."""
    dataset_csv = rf"/hpc2hdd/home/qwang650/project/yutianjiang/stmodel/dataset/indicator/output/{args.city}_indicator_metadata.csv"
    data = pd.read_csv(dataset_csv)

    # split dataset into train, val, test
    data = data.sample(frac=1).reset_index(drop=True)  # shuffle rows
    train_data = data[: int(len(data) * args.train_dataset_ratio)].reset_index(
        drop=True
    )
    val_data = data[int(len(data) * args.train_dataset_ratio) :].reset_index(drop=True)
    mean = np.mean(train_data[args.indicator])
    std = np.std(train_data[args.indicator])

    train_transform = get_train_transform(224)
    val_transform = get_val_transform(224)

    # create datasets
    train_dataset = IndicatorDataset(
        args.city, train_data, args.indicator, train_transform, mean, std, False
    )
    val_dataset = IndicatorDataset(
        args.city, val_data, args.indicator, val_transform, mean, std, False
    )

    if args.test_file is not None:
        test_data = pd.read_csv(args.test_file)
        test_dataset = IndicatorDataset(
            args.city, test_data, args.indicator, val_transform, mean, std, True
        )
        return train_dataset, val_dataset, test_dataset, mean, std
    else:
        return train_dataset, val_dataset, None, mean, std


def get_train_transform(size, s=1):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([
        transforms.CenterCrop(size=size),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomApply([color_jitter], p=0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return data_transforms

def get_val_transform(size):
    data_transforms = transforms.Compose([
        transforms.CenterCrop(size=size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return data_transforms