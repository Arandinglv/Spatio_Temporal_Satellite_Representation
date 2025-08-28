import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import datasets

from utils.data_aug import GaussianBlur
from utils.data_aug import CutOut
from utils.data_aug import Sobel
from utils.view_generator import ContrastiveLearningViewGenerator
from datasets.spatio_temporal_dataset import SpatioTemporalDataset
from datasets.STlabelpositive_dataset import STlabelpositive
from exceptions.exceptions import InvalidDatasetSelection
from datasets.STlabel_dataset import STlabel
from datasets.STlabel_spatio_same_dataset import STlabel_spatio_same


class ContrastiveLearningDataset:
    def __init__(self, root_folder, num_neg):
        self.root_folder = root_folder
        self.num_neg = num_neg
        
    @staticmethod
    # get_simclr_pipeline_transform是静态方法, 
    # 不需要实例化ContrastiveLearningDataset就能调用
    def get_simclr_pipeline_transform(size, s=1):
        """
        包含resize, horizonFlip, cutout, 
        sobel, gray, gaussianBlur, rotate, 
        colorJitter, Normalize
        """
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(p=0.4), 
            transforms.RandomApply([color_jitter], p=0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(kernel_size=7)], p=0.3), 
            transforms.RandomApply([CutOut(n_holes=3, length=5)], p=0.4), 
            # transforms.RandomApply([transforms.RandomRotation(30)], p=0.1),
            # transforms.RandomApply([Sobel()], p=0.2),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        return data_transforms
    
    def get_dataset(self, name, n_views=2):
        # 有效的数据集
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(
                self.root_folder, train=True,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(32),
                    n_views
                ),
                download=True
            ),
            'stl10': lambda: datasets.STL10(
                self.root_folder, split='unlabeled',
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(96),
                    n_views
                ),
                download=True
            ),
            'SpatioTemporalDataset': lambda:SpatioTemporalDataset(
                self.root_folder, 
                num_neg=self.num_neg, 
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(224),
                    n_views=2
                ),
                transform_neg=self.get_simclr_pipeline_transform(224), 
            ),
            # SpatioTemporalDataset是不带label的
            
            'STlabelpositive': lambda:STlabelpositive(
                self.root_folder, 
                num_neg=self.num_neg, 
                transform=self.get_simclr_pipeline_transform(224), 
                transform_neg=self.get_simclr_pipeline_transform(224), 
            ), 
            # STlabelpositive 带label, 同一个label不同年份作为正样本
            # (对于 spatial temporal spatial-temporal都是如此)
            
            'STlabel': lambda:STlabel(
                self.root_folder, 
                num_neg=self.num_neg,
                transform = ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(224),
                    n_views=2
                ),
                transform_neg=self.get_simclr_pipeline_transform(224)
            ), 
            
            'STlabel_spatio_same': lambda:STlabel_spatio_same(
                self.root_folder, 
                num_neg=self.num_neg,
                transform = ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(224),
                    n_views=2
                ),
                transform_neg=self.get_simclr_pipeline_transform(224)
            ), 
            
            
        }  
        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection(f"Dataset '{name}' is not available. Available datasets: {list(valid_datasets.keys())}")
        else: 
            return dataset_fn() 
        
        
        