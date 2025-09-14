import torch
import torch.nn as nn
import torchvision.models as models

class ResNetSimCLR(nn.Module):
    def __init__(self, base_model='resnet18', out_dim=128):
        super(ResNetSimCLR, self).__init__()
        
        self.encoder = self._get_base_model(base_model)
        # TODO: 添加空间注意力？
        if base_model.startswith('resnet'):
            dim_mlp = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()  # 去掉网络的最后一层
        elif base_model.startswith('vit'):
            dim_mlp = self.encoder.heads.head.in_features
            self.encoder.heads.head = nn.Identity()  # 去掉网络的最后一层
        
        self.projection = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), 
            nn.ReLU(), 
            nn.Linear(dim_mlp, dim_mlp * 2), 
            nn.ReLU(), 
            nn.Linear(dim_mlp * 2, out_dim)
        )
        
    def _get_base_model(self, base_model):
        if base_model == 'resnet18':
            return models.resnet18(weights=None)
        elif base_model == 'resnet50':
            return models.resnet50(weights=None)
        elif base_model == 'vit_b_16':
            return models.vit_b_16(weights=None)
        else:
            raise NotImplementedError(f"Base model {base_model} not supported") 
        
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return z
    
    
