"""
这个dataset是继承于 experiments/dataset_class/dataset.py
位置上的负样本只有本年份其他位置的tiles

时间负样本构建直接通过num_neg的数量来决定
"""
import os
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SpatioTemporalDataset(Dataset):
    """
    数据集文件夹中, city下包含jpg, jpg下包含不同年份
    """
    
    def __init__(self, root_folder, num_neg=None, transform=None, transform_neg=None):
        """
        Args:
            root_folder (str): 数据集根目录
            transform (callable, optional): 数据增强方法，用于 anchors，应该返回一个列表
            transform_neg (callable, optional): 数据增强方法，用于 temporal_negatives，应该返回一个 Tensor
            num_tempo_neg_aug (int, optional): 每个时间负样本的增强次数
        """
        
        self.root_folder = root_folder
        self.transform = transform
        self.transform_neg = transform_neg
        # self.num_tempo_neg_aug = num_tempo_neg_aug
        self.num_neg = num_neg
        self.image_dict = self._organize_images()
        self.locations = list(self.image_dict.keys())
        
    def _organize_images(self):
        """
        根据城市和经纬度关联不同年份的图像, 便于负样本的采集
        return: 
            dict: {location_key: {year: image_path, ...}, ...}
        """
        # todo: 这里数据集每个城市都包含相同数量的年份图像, 
        # todo: 后面需要调整成相同城市同一位置会包含不同数量年份的数据, 
        # todo: 通过数据增强等操作得到相同数量的图像
        # todo: 所有图像应该放在同一个文件夹下
        # todo: 所有地理位置应该放在同一个csv下
        image_dict = {}
        years = [d for d in os.listdir(self.root_folder) 
                if os.path.isdir(os.path.join(self.root_folder, d))]
        for year in years:
            year_path = os.path.join(self.root_folder, year)
            csv_files = [f for f in os.listdir(year_path) if f.endswith('_metadata.csv')]
            if len(csv_files) == 0:
                raise ValueError(f"No metadata file found for year {year}")
            csv_path = os.path.join(year_path, csv_files[0])
            metadata = pd.read_csv(csv_path)
            # metadata中包含: tile_name,left_top_lon,left_top_lat,right_bottom_lon,right_bottom_lat
            for _, row in metadata.iterrows():
                tile_name = row['tile_name']
                name_parts = tile_name.split('_')
                city = name_parts[0]
                year = name_parts[1]
                left_top_lon = name_parts[2]
                left_top_lat = name_parts[3]
                right_bottom_lon = name_parts[4]
                right_bottom_lat = name_parts[5].split('.jpg')[0]  # 去除扩展名
                location_key = f"{city}_{left_top_lon}_{left_top_lat}_{right_bottom_lon}_{right_bottom_lat}"    
                
                if location_key not in image_dict:
                    image_dict[location_key] = {}
                image_path = os.path.join(year_path, tile_name)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                image_dict[location_key][year] = image_path
                
        return image_dict


    def __len__(self):
        return len(self.locations)
    
    def __getitem__(self, idx):
        """
        时间和空间上的对比学习正样本对是一致的, 但是负样本不同:
        空间负样本: 该年份其他位置的tiles
        时间负样本: 该地点的其他年份数据
        
        获取一个样本，包括 anchors 图像和 temporal_negatives

        Args:
            idx (int): 样本索引

        Returns:
            dict: {
                "anchors": list of Tensors, [n_views, C, H, W],
                "temporal_negatives": Tensor, [num_neg, C, H, W]
            }
        """
        location_key = self.locations[idx]
        years_available = list(self.image_dict[location_key].keys())
        # 随机选择一个年份作为anchor
        anchor_year = random.choice(years_available)
        anchor_image_path = self.image_dict[location_key][anchor_year]
        anchor_image = Image.open(anchor_image_path).convert('RGB')
        
        if self.transform:
            anchors = self.transform(anchor_image)  # list of n_views tensors
            if not isinstance(anchors, list):
                raise TypeError(f"Expected anchors to be a list, but got {type(anchors)}")
            for view in anchors:
                if not isinstance(view, torch.Tensor):
                    raise TypeError(f"Expected anchor views to be tensors, but got {type(view)}")
        else:
            raise ValueError("Transform must be provided to convert images to tensors.")
        
        # 获取时间负样本
        temporal_negatives = []
        negative_years = [y for y in years_available if y != anchor_year]
        
        for _ in range(self.num_neg):
            neg_year = random.choice(negative_years)
            neg_image_path = self.image_dict[location_key][neg_year]
            neg_image = Image.open(neg_image_path).convert('RGB')
            if self.transform_neg:
                neg_image = self.transform_neg(neg_image)
                # 调试信息
                if not isinstance(neg_image, torch.Tensor):
                    raise TypeError(f"Expected neg_augmented to be a Tensor, but got {type(neg_image)}")
                temporal_negatives.append(neg_image)
            else:     
                raise ValueError("Transform_neg must be provided to convert images to tensors.")
        
        if len(temporal_negatives) == 0:
            raise ValueError("No temporal negatives found for this sample.")   
        
        temporal_negatives = torch.stack(temporal_negatives, dim=0)  # [num_neg, C, H, W]  
        
        
        # # 以下得到的负样本数量是 year *   num_tempo_neg_aug
        # for neg_year in negative_years:
        #     neg_image_path = self.image_dict[location_key][neg_year]
        #     neg_image = Image.open(neg_image_path).convert('RGB')  
        #     if self.transform_neg:
        #         for _ in range(self.num_tempo_neg_aug):
        #             neg_augmented = self.transform_neg(neg_image)
        #             # 调试信息
        #             if not isinstance(neg_augmented, torch.Tensor):
        #                 raise TypeError(f"Expected neg_augmented to be a Tensor, but got {type(neg_augmented)}")
        #             temporal_negatives.append(neg_augmented)
        #     else:     
        #         raise ValueError("Transform_neg must be provided to convert images to tensors.")
        # 
        # if len(temporal_negatives) == 0:
        #     raise ValueError("No temporal negatives found for this sample.")
        # 
        # temporal_negatives = torch.stack(temporal_negatives, dim=0)  # [num_neg, C, H, W]
        
        
        # 调试信息
        if not isinstance(temporal_negatives, torch.Tensor):
            raise TypeError(f"Expected temporal_negatives to be a tensor, but got {type(temporal_negatives)}")
        if temporal_negatives.dim() != 4:
            raise ValueError(f"Expected temporal_negatives to have 4 dimensions, but got {temporal_negatives.dim()}")
    
        # # 可选：打印前几个样本信息
        # if idx < 5:
        #     print(f"Sample {idx}:")
        #     print(f"  Anchors: {[view.shape for view in anchors]}")
        #     print(f"  Temporal_negatives: {temporal_negatives.shape}")
    
        return {
            "anchors": anchors,  # list of n_views tensors [n_views, C, H, W]
            "temporal_negatives": temporal_negatives  # [num_neg, C, H, W]
        }