import random
import torch
from torch.utils.data import Sampler

class YearBatchSampler(Sampler):
    """
    每次产出一个 batch:
      1) 随机选一个 year
      2) 在所有 location 索引中选出 batch_size 个
    如此反复 num_batches 次。
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        """
        Args:
            dataset: 我们上面定义的 STlabelYearDataset 实例
            batch_size (int): batch 大小
            shuffle (bool): 是否在每个 epoch 打乱 location 索引
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # dataset.locations 的长度
        self.num_locations = len(dataset.locations)
        # 这里决定一共产出多少个 batch
        self.num_batches = self.num_locations // self.batch_size

        # 记录所有年份（比如 [2010,2011,...,2020]）
        self.all_years = self.dataset.all_years

    def __iter__(self):
        # 首先把所有 location 的索引打乱（或者保持不变）
        indices = list(range(self.num_locations))
        if self.shuffle:
            random.shuffle(indices)

        start = 0
        for _ in range(self.num_batches):
            # 1) 随机选一个 year
            chosen_year = random.choice(self.all_years)
            # 2) 取出 batch_size 个 location 的子集
            batch_indices = indices[start : start + self.batch_size]
            start += self.batch_size

            # 把 (year, location_idx) 二元组拼成一个 list
            yield [(chosen_year, loc_idx) for loc_idx in batch_indices]

    def __len__(self):
        """
        返回本 epoch 能产生的 batch 数量
        """
        return self.num_batches
