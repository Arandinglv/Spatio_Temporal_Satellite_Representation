import os
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import ast

class STlabelScoreGroupDataset(Dataset):
    
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
        self.years = sorted([str(year) for year in os.listdir(self.root_folder) 
                if os.path.isdir(os.path.join(self.root_folder, year))])
        self.image_dict = self._organize_images()
        self.locations = self._filter_locations()
        print(f"Valid locations count: {len(self.locations)}")

                
                
    def _create_location_key_from_tilename(self, tile_name):
        """
        根据tile_name: 
        {city}_{year}_{left_top_lon}_{left_top_lat}_{right_bottom_lon}_{right_bottom_lat}.jpg
        得到相应的location key
        """
        base_name = os.path.basename(tile_name)
        city, year, left_top_lon, \
            left_top_lat, right_bottom_lon, right_bottom_lat = base_name.split('_')
        return f"{city}_{left_top_lon}_{left_top_lat}_{right_bottom_lon}_{right_bottom_lat}"
    
    
    def _calculate_neighbours(self, metadata_df, tile_name):
        # 当前tile的经纬度，直接使用index查找
        location = metadata_df.loc[tile_name]
        # 邻居信息
        lon = [location["left_top_lon"], location["right_bottom_lon"]]
        lat = [location["left_top_lat"], location["right_bottom_lat"]]
        neighbours_df = metadata_df[
            (metadata_df["left_top_lon"].isin(lon)) &
            (metadata_df["left_top_lat"].isin(lat))
        ]  
        # 打印出来的是带有年份的.
        neighbour_list = neighbours_df.index.tolist()
        if len(neighbour_list) <= 1:
            return []
            
        return neighbour_list    
    
    
    def _filter_locations(self):
        """
        把image_dict中label全为0的记录在一个labels_all_zero set里
        然后把不全为0的放在valid_locations set里
        如果location_key的neighbours中包含了labels_all_zero里的location，
        那么就要删掉这个location_key
        但要保持image_dict的完整性，不删除边缘信息
        """
        # 找出所有label全为0的location
        labels_all_zero = set()
        valid_locations = set()
        
        for location_key, years_data in self.image_dict.items():
            labels = [year_data['label'] for year_data in years_data.values()]
            if all(label == 0 for label in labels):
                labels_all_zero.add(location_key)
            else:
                valid_locations.add(location_key)
        
        # 检查valid_locations中的每个location，如果其邻居中包含全为0的location，则移除
        locations_to_remove = set()
        for location_key in valid_locations:
            first_year = next(iter(self.image_dict[location_key]))
            neighbours = self.image_dict[location_key][first_year]['neighbours']
            
            if not neighbours:
                locations_to_remove.add(location_key)
                continue
                
            for neighbour in neighbours:
                neighbour_location_key = self._create_location_key_from_tilename(neighbour)
                # 如果邻居中有label全为0的，或者邻居不在image_dict中，则移除该location
                if (neighbour_location_key in labels_all_zero or 
                    neighbour_location_key not in self.image_dict):
                    locations_to_remove.add(location_key)
                    break
        
        # 从 valid_locations 中移除要删除的location
        valid_locations -= locations_to_remove
        
        return list(valid_locations)
        
    
        
    def _organize_images(self):
        """
        return: 
            {
                {location_key: {year: 
                                    "path": image_path,
                                    "label": label, 
                                    "ratios": {
                                            2010: [], 
                                            2011: [],
                                            ...
                                            2020: [],     
                                    }, 
                                    "neighbours": [neighbour_0（自身）,
                                                   neighbour_1,
                                                   neighbour_2,
                                                   neighbour_3]
                                }
            }
        neighbour都是location_key的形式
        {city}_{left_top_lon}_{left_top_lat}_{right_bottom_lon}_{right_bottom_lat}
        """
        image_dict = {}
        
        for year in self.years: 
            year_path = os.path.join(self.root_folder, year)
            csv_file = [f for f in os.listdir(year_path) if f.endswith('_metadata.csv')]
            if len(csv_file) == 0:
                raise ValueError(f"No metadata file found for year {year}") 
            csv_path = os.path.join(year_path, csv_file[0])
            metadata = pd.read_csv(csv_path, index_col=0)
            
            for tile_name, row in metadata.iterrows():
                # tile_name现在是索引，row是数据
                year = str(row["year"])
                location_key = self._create_location_key_from_tilename(tile_name)
                # 创建相应的location字典，用于填充内容
                if location_key not in image_dict:
                    image_dict[location_key] = {} 
                image_path = os.path.join(year_path, tile_name)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                if year not in image_dict[location_key]:
                    image_dict[location_key][year] = {
                        'path': os.path.join(year_path, tile_name), 
                        'label': int(row["label"]), 
                        'ratios': {}, 
                        'neighbours': {}
                    }

                # 填充每一年的ratio
                for col in metadata.columns:
                    if col.isdigit():
                        year_target = str(col)
                        row_col_content = row[col]
                        if pd.isna(row_col_content) or row_col_content == "":
                            ratio_list= []
                        else:
                            ratio_list = ast.literal_eval(row_col_content)  # "[0.12, 0.08, ...]" => list
                        image_dict[location_key][year]["ratios"][year_target] = ratio_list
                
                # # 填充neighbours
                # neighbour_list = ast.literal_eval(row['neighbours'] if isinstance(row['neighbours'], str) else [])
                # neighbour_list = [self._create_location_key_from_tilename(neighbour) for neighbour in neighbour_list]
                # image_dict[location_key][year]['neighbours'] = neighbour_list
                
                # 用前面的_calculate_neighbours来构建
                neighbour_list = self._calculate_neighbours(metadata, tile_name)
                # neighbour_list = [self._create_location_key_from_tilename(neighbour) for neighbour in neighbour_list]
                image_dict[location_key][year]['neighbours'] = neighbour_list
                
        return image_dict
    
    
    def __len__(self):
        return len(self.locations)
    
    def __getitem__(self, idx):
        """
        return: 
            {
                "anchors": [n_views, C, H, W],    # list of tensors
                "temporal_negatives": [num_neg, C, H, W],
                "temporal_negatives_ratios": [num_neg, label_num],
            }
        """
        location_core = self.locations[idx]
        # years_available = list(self.image_dict[location_key].keys())
        
        # # 1) anchor 
        anchor_year = random.choice(self.years)
        # print(anchor_year)
        location_key_quarter_path = self.image_dict[location_core][anchor_year]['neighbours']
        
        dict_list = []

        for location_key_path in location_key_quarter_path:
            # 1) anchor
            location_key = self._create_location_key_from_tilename(location_key_path)
            anchor_image_path = self.image_dict[location_key][anchor_year]['path']
            anchor_image_label = self.image_dict[location_key][anchor_year]['label']
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
            
            # 2) temporal negatives
            temporal_negatives = []
            negative_years = [str(year) for year in self.years if year != anchor_year]
            temporal_negatives_ratios = [] 
            
            for _ in range(self.num_neg):
                # 找到所有与 anchor 标签不同的年份
                valid_neg_years = []
                for year in negative_years:
                    if self.image_dict[location_key][year]['label'] != anchor_image_label:
                        valid_neg_years.append(year)
                
                if not valid_neg_years:
                    # 如果没有有效的负样本年份，跳过
                    continue
                    
                neg_year = random.choice(valid_neg_years)
                neg_image_path = self.image_dict[location_key][neg_year]['path']
                neg_image = Image.open(neg_image_path).convert('RGB')
                if self.transform_neg:
                    neg_image = self.transform_neg(neg_image)
                    # 调试信息
                    if not isinstance(neg_image, torch.Tensor):
                        raise TypeError(f"Expected neg_augmented to be a Tensor, but got {type(neg_image)}")
                    temporal_negatives.append(neg_image)
                else:     
                    raise ValueError("Transform_neg must be provided to convert images to tensors.")
                
                ratio_dict = self.image_dict[location_key][anchor_year]['ratios']
                ratio_list = ratio_dict[str(neg_year)]
                
                ratio_tensor = torch.tensor(ratio_list, dtype=torch.float32) 
                temporal_negatives_ratios.append(ratio_tensor) 
            

            dict_list.append({
            "anchors": anchors,  # list of Tensors, shape: [n_views, C, H, W]
            "temporal_negatives": temporal_negatives,  # list of tensors: [num_neg, C, H, W]
            "temporal_negatives_ratios": temporal_negatives_ratios,  # list of tensors: [num_neg, label_num]
            })
            # 这里一次性送入四张图

    
        return dict_list
        
    # @property
    # def num_valid_locations(self):
    #     return len(self.locations)

