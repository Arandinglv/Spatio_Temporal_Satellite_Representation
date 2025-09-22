import os.path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class IndicatorDataset(Dataset):
    def __init__(
        self,
        city = "Guangzhou",
        df_data = None,
        indicator = "CO2",
        transform = None,
        mean = 1.0,
        std = 1.0,
        is_test = False
    ):
        super().__init__()

        self.transform = transform  # image transform for CoCa

        self.img_paths = []
        # self.img_tensors = []
        self.y = []


        for idx, row in df_data.iterrows():
            img_name = row["satellite_img_name"]
            img_path = os.path.join(rf"/hpc2hdd/home/qwang650/project/yutianjiang/stmodel/dataset/{city}/2020", img_name)
            self.img_paths.append(img_path)
            if is_test:  # test set no real indicator value
                self.y.append(0.0)
            else:
                self.y.append((row[indicator] - mean) / std)


    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        _im = Image.open(img_path).convert("RGB")
        # im = transform(im).unsqueeze(0)  # [1, 3, 224, 224]
        _im = self.transform(_im)  # [3, 224, 224]
        return _im, np.float32(self.y[index])

