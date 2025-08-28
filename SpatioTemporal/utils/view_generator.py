from torchvision import transforms
from utils.data_aug import GaussianBlur  # 确保你有这个模块

class ContrastiveLearningViewGenerator:
    """生成多个随机视图作为正样本"""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views
    
    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]