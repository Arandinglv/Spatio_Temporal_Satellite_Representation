"""
这个dataset是继承于 experiments/dataset_class/dataset.py
位置上的负样本只有本年份其他位置的tiles

图片都带上了label

没用上transform，只用上了transform_neg
"""
import os
from numpy import positive
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class STlabelpositive(Dataset):
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
        self.locations = self._filter_locations()
        
    def _organize_images(self):
        """
        根据城市和经纬度关联不同年份的图像, 便于负样本的采集
        return: 
            dict: {location_key: {year: 
                                        "path": image_path, 
                                        "label": label
                                    }, 
                                ...}
        """
        # todo: 这里数据集每个城市都包含相同数量的年份图像, 
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
                # name_parts = tile_name.split('_')
                year = row["year"]
                location_key = self._create_location_key(row)
                if location_key not in image_dict:
                    image_dict[location_key] = {}
                image_path = os.path.join(year_path, tile_name)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                image_dict[location_key][year] = {
                    'path': os.path.join(year_path, row["tile_name"]), 
                    'label': int(row["label"])
                }
        return image_dict
    
    
    def _create_location_key(self, row):
        """
        根据row创建location_key
        """
        return f"{row['city']}_{row['left_top_lon']}_{row['left_top_lat']}_" \
               f"{row['right_bottom_lon']}_{row['right_bottom_lat']}"
            
            
    def _filter_locations(self):
        """
        过滤掉该区域的所有年份均为label=0的location
        """
        valid_locations = []
        for location_key, years in self.image_dict.items():
            # 过滤掉所有年份均为label=0的location
            labels = [pathAndyear['label'] for pathAndyear in years.values()]
            if not all(label == 0 for label in labels):
                valid_locations.append(location_key)
            
        return valid_locations
        

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
        anchor_image_path = self.image_dict[location_key][anchor_year]['path']
        anchor_image_label = self.image_dict[location_key][anchor_year]['label']
        anchor_image = Image.open(anchor_image_path).convert('RGB')
        
        if self.transform_neg:
            anchor = self.transform(anchor_image)  
        else:
            raise ValueError("Transform must be provided to convert images to tensors.")
        
        # 获取时间正样本（从label一致的图像中选取）
        positive_year = random.choice([year for year in years_available if self.image_dict[location_key][year]['label'] == anchor_image_label])
        positive_image_path = self.image_dict[location_key][positive_year]['path']
        positive_image = Image.open(positive_image_path).convert('RGB')
        if self.transform_neg:
            positive_image = self.transform(positive_image)
        else:     
            raise ValueError("Transform_neg must be provided to convert images to tensors.")
        
        # 把anchor和positive拼接在一起得到list of tensors: 2, c, h, w
        anchors = [anchor, positive_image]
        
        # 获取时间负样本
        temporal_negatives = []
        negative_years = [y for y in years_available if y != anchor_year]
        
        for _ in range(self.num_neg):
            neg_year = random.choice(negative_years)
            neg_image_path = self.image_dict[location_key][neg_year]['path']
            neg_image_label = self.image_dict[location_key][neg_year]['label']
            # 如果negative和anchor的标签一致, 则需要重新选择一个负样本
            while neg_image_label == anchor_image_label:
                neg_year = random.choice(negative_years)
                neg_image_path = self.image_dict[location_key][neg_year]['path']
                neg_image_label = self.image_dict[location_key][neg_year]['label']
                
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
        
        # 调试信息
        if not isinstance(temporal_negatives, torch.Tensor):
            raise TypeError(f"Expected temporal_negatives to be a tensor, but got {type(temporal_negatives)}")
        if temporal_negatives.dim() != 4:
            raise ValueError(f"Expected temporal_negatives to have 4 dimensions, but got {temporal_negatives.dim()}")

    
        return {
            "anchors": anchors,  # list of n_views tensors [2, C, H, W]
            "temporal_negatives": temporal_negatives  # [num_neg, C, H, W]
        }
        