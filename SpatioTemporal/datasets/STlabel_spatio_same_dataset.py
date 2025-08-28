import os
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class STlabel_spatio_same(Dataset):
    """
    与原先 STlabel(Dataset) 类似，但不再在 __getitem__ 里随机挑选 year，
    而是通过 (year, location_idx) 的形式由外部传入。
    """

    def __init__(self, root_folder, num_neg=None, transform=None, transform_neg=None):
        """
        Args:
            root_folder (str): 数据集根目录，类似 '/data/xxx/jpg'
            num_neg (int): 负样本数量
            transform (callable): 用于 anchors 的数据增强，返回 list of Tensors
            transform_neg (callable): 用于 temporal_negatives 的数据增强，返回 Tensor
        """
        self.root_folder = root_folder
        self.num_neg = num_neg
        self.transform = transform
        self.transform_neg = transform_neg

        # 根据你的逻辑组织数据
        self.image_dict = self._organize_images()
        # 过滤掉所有年份都为 label=0 的 location
        self.locations = self._filter_locations()

        # 这里收集所有出现过的年份，方便 Sampler 随机选择
        # 假设 year 在 CSV 里是 int，如果是 string 则请根据情况处理
        all_years = set()
        for loc, year_dict in self.image_dict.items():
            for y in year_dict.keys():
                all_years.add(y)
        self.all_years = sorted(list(all_years))

    def _organize_images(self):
        """
        根据城市和经纬度，将不同年份的图像关联到一起，构建:
        image_dict[location_key][year] = {
            "path": <image_path>, 
            "label": <label>
        }
        """
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
            for _, row in metadata.iterrows():
                tile_name = row['tile_name']
                location_key = self._create_location_key(row)
                if location_key not in image_dict:
                    image_dict[location_key] = {}
                
                image_path = os.path.join(year_path, tile_name)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                
                image_dict[location_key][row["year"]] = {
                    'path': image_path,
                    'label': int(row["label"])
                }
        return image_dict

    def _create_location_key(self, row):
        """
        根据 row 创建 location_key
        例如: city_lefttoplon_lefttoplat_rightbottomlon_rightbottomlat
        """
        return f"{row['city']}_{row['left_top_lon']}_{row['left_top_lat']}_" \
               f"{row['right_bottom_lon']}_{row['right_bottom_lat']}"

    def _filter_locations(self):
        """
        过滤掉该区域所有年份均为 label=0 的location
        """
        valid_locations = []
        for location_key, years_info in self.image_dict.items():
            labels = [info['label'] for info in years_info.values()]
            if not all(label == 0 for label in labels):
                valid_locations.append(location_key)
        return valid_locations

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, item):
        """
        item 是一个 (year, location_idx) 二元组，
        year: 当批次统一的年份(由 Sampler 指定)
        location_idx: self.locations 里的索引
        """
        year, loc_idx = item
        location_key = self.locations[loc_idx]

        # ========== anchor 图像 ==========
        anchor_info = self.image_dict[location_key][year]
        anchor_image_path = anchor_info['path']
        anchor_label = anchor_info['label']
        anchor_image = Image.open(anchor_image_path).convert('RGB')

        # transform 得到 anchors（注意你的 transform 里通常会返回一个 list of Tensor）
        if self.transform is None:
            raise ValueError("transform must be provided for anchor images.")
        anchors = self.transform(anchor_image)
        if not isinstance(anchors, list):
            raise TypeError(f"Expected anchors to be a list of Tensors, got {type(anchors)}")

        # ========== 负样本：从同一个 location 但是其它年份里选 ==========
        temporal_negatives = []
        negative_years = [y for y in self.image_dict[location_key].keys() if y != year]

        for _ in range(self.num_neg):
            neg_year = random.choice(negative_years)
            neg_info = self.image_dict[location_key][neg_year]
            neg_label = neg_info['label']
            # 如果负样本和 anchor 的 label 一致，就换一个重选
            while neg_label == anchor_label:
                neg_year = random.choice(negative_years)
                neg_info = self.image_dict[location_key][neg_year]
                neg_label = neg_info['label']
            neg_image_path = neg_info['path']
            neg_image = Image.open(neg_image_path).convert('RGB')

            if self.transform_neg is None:
                raise ValueError("transform_neg must be provided for negative images.")
            neg_image_tensor = self.transform_neg(neg_image)

            if not isinstance(neg_image_tensor, torch.Tensor):
                raise TypeError(f"Expected negative image transform to return a Tensor, got {type(neg_image_tensor)}")

            temporal_negatives.append(neg_image_tensor)

        if len(temporal_negatives) == 0:
            raise ValueError("No temporal negatives found for this sample.")

        temporal_negatives = torch.stack(temporal_negatives, dim=0)  # shape: [num_neg, C, H, W]

        return {
            "anchors": anchors,                  # list of Tensors, shape each is [C, H, W]
            "temporal_negatives": temporal_negatives  # [num_neg, C, H, W]
        }
