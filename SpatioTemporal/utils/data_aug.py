# data_aug/gaussian_blur.py
import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
import torch.nn.functional as F
import random
import torchvision
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import to_pil_image



class GaussianBlur(object):
    """
    在CPU上对单个图像进行高斯模糊
    copy simclr_pytorch的gaussian_blur
    """
    def __init__(self, kernel_size):
        radius = kernel_size // 2
        kernel_size = radius * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radius

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radius),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img
    

class CutOut(object):
    """Randomly mask out one or more patches from an image."""
    def __init__(self, n_holes=1, length=16, p=1.0):
        """
        Args:
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
            p (float): Probability of applying this transformation.
        """
        self.n_holes = n_holes
        self.length = length
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cutout.
        Returns:
            PIL Image or Tensor: CutOut applied image.
        """
        # 如果想对 PIL Image 做 CutOut，需要转 tensor 后再转回去
        # 如果你本身的数据增强管线已经是 Tensor，这里直接处理 Tensor 即可
        if random.random() > self.p:
            return img  # 不执行 CutOut
        
        # 如果是 PIL Image，需要先转成 tensor
        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img)  # [C, H, W]

        _, h, w = img.shape

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(np.clip(y - self.length // 2, 0, h))
            y2 = int(np.clip(y + self.length // 2, 0, h))
            x1 = int(np.clip(x - self.length // 2, 0, w))
            x2 = int(np.clip(x + self.length // 2, 0, w))

            img[:, y1:y2, x1:x2] = 0
            

        img = to_pil_image(img)  

        return img
    
class Sobel(object):
    def __init__(self, p):
        # Define Sobel kernels in 2D, shape -> (1,1,3,3) so they can be applied via conv2d
        sobel_x = [[-1., 0., 1.],
                   [-2., 0., 2.],
                   [-1., 0., 1.]]

        sobel_y = [[-1., -2., -1.],
                   [ 0.,  0.,  0.],
                   [ 1.,  2.,  1.]]

        self.sobel_x = torch.tensor(sobel_x).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)
        self.sobel_y = torch.tensor(sobel_y).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)

        self.p = p

    def __call__(self, img):
        """
        Args:
            img

        Returns:
            PIL.Image
        """
        if random.random() > self.p:
            return img

        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img)  


        C, H, W = img.shape
        edge_channels = []
        for c in range(C):
            # [1, 1, H, W] 
            channel = img[c].unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

            grad_x = F.conv2d(channel, self.sobel_x, padding=1)  # (1,1,H,W)
            grad_y = F.conv2d(channel, self.sobel_y, padding=1)  # (1,1,H,W)

            grad_mag = torch.sqrt(grad_x**2 + grad_y**2)  # (1,1,H,W)

            grad_mag = grad_mag.squeeze(0).squeeze(0)   # (H, W)
            edge_channels.append(grad_mag)  

        edges = torch.stack(edge_channels, dim=0)  # [C, H, W]
        
        return TF.to_pil_image(edges)