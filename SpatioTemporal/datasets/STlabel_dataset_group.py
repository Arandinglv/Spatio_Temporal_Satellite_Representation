import os
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import ast
from scipy.stats import *

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
        self.years = [str(year) for year in os.listdir(self.root_folder) 
                if os.path.isdir(os.path.join(self.root_folder, year))]
        self.image_dict = self._organize_images()
        self.locations = self._filter_locations()
        print(len(self.locations))

                
                
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
        # 当前tile的经纬度
        location = metadata_df.loc[tile_name]
        # 邻居信息
        lon = [location["left_top_lon"], location["right_bottom_lon"]]
        lat = [location["left_top_lat"], location["right_bottom_lat"]]
        neighbours_df = metadata_df[
            (metadata_df["left_top_lon"].isin(lon)) &
            (metadata_df["left_top_lat"].isin(lat))
        ]  
        # 打印出来的是带有年份的.
        neighbour_list = neighbours_df["tile_name"].tolist()
        neighbour_list = [self._create_location_key_from_tilename(i) for i in neighbour_list]
        if len(neighbour_list) < 4:
            return []
        return neighbour_list    
    
    def _filter_label_zero(self, all_locations, metadata):
        allzero_location = set()
        for location_key in all_locations:
            # 过滤掉所有年份为label=0的location
            labels = [content['label'] for content in self.locations[location_key].values()]
            if all(label == 0 for label in labels):
                allzero_location.add(location_key)

        # 过滤出邻居含有全0
        invalid_locations = set()
        for location_key in all_locations:
            neighbours = self.image_dict[location_key]['neighbours']
            for neighbour in neighbours:
                if neighbour in allzero_location:
                    invalid_locations.add(location_key)
        return invalid_locations


    def _filter_locations(self):
        """
        过滤掉所有年份均为label=0的location，和不合规的group
        """
        year_path = os.path.join(self.root_folder, self.years[0])
        csv_file = [f for f in os.listdir(year_path) if f.endswith('_metadata.csv')]
        if len(csv_file) == 0:
            raise Exception("No metadata.csv found")
        csv_file = os.path.join(year_path, csv_file[0])
        metadata = pd.read_csv(csv_file, index_col=0)

        all_locations = {location_key for location_key, _ in self.image_dict.items()}
        invalid_locations = self._filter_label_zero(all_locations, metadata)

        print(f"Invalid locations due to label zero: {len(invalid_locations)}")
        valid_locations = all_locations - invalid_locations

        # 去除边界
        valid_locations = set()
        for location_key in valid_locations:
            neighbours = self.image_dict[location_key]['neighbours']
            if len(neighbours) != 4:
                invalid_locations.add(location_key)
        print(f"Invalid locations due to boundary: {len(invalid_locations)}")
        valid_locations = list(valid_locations-invalid_locations)

        print(f"Valid locations {len(valid_locations)}")
        return valid_locations

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
            metadata = pd.read_csv(csv_path)
            
            for _, row in metadata.iterrows():
                tile_name = row['tile_name']
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
                        'path': os.path.join(year_path, row["tile_name"]), 
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
            location_key = location_key_path
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
            temporal_negatives_ratios = []

            possible_negatives = [year for year in self.years if self.image_dict[location_key][year]['label'] != anchor_image_label]

            # TODO: sigma 变换
            #
            # current_sigma = 0.3 + (max_epoch -  current_epoch)

            a = (0-0) / current_sigma
            b = (1-0) / current_sigma

            possible_negatives_sorted = sorted(possible_negatives, key=lambda y:abs(int(y)-int(anchor_year)))

            if self.current_epoch <= self.max_epoches:
                r = truncnorm.rvs(a,b, loc = 0, scale = current_sigma, size=self.nunm_neg)
                neg_indexs = (r*len(possible_negatives_sorted)).astype(int)
            else:
                neg_indexs = sorted(np.random.choice(range(len(possible_negatives)), size=self.nunm_neg, replace=True))

            for neg_index in neg_indexs:
                neg_year = possible_negatives_sorted[neg_index]
                neg_image_path = self.image_dict[location_key][neg_year]['path']
                neg_image = Image.open(neg_image_path).convert('RGB')

                if self.transform_neg:
                    neg_image = self.transform_neg(neg_image)
                    if not isinstance(neg_image, torch.Tensor):
                        raise TypeError(f"Expected anchor views to be tensors, but got {type(neg_image)}")
                    temporal_negatives.append(neg_image)
                else:
                    raise ValueError("Transform must be provided to convert images to tensors.")

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

