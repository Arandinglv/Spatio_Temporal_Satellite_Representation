import torch
import torch.nn as nn
import torchvision.models as models


class SoftCon(nn.Module):
    def __init__(self, base_model='resnet50', out_dim=128):
        super().__init__()
        self.encoder = self._get_base_model(base_model)
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        return self.encoder(x)

    def _get_base_model(self, base_model):
        if base_model == 'resnet18':
            return models.resnet18(weights=None)
        elif base_model == 'resnet50':
            return models.resnet50(weights=None)
        elif base_model == 'vit_b_16':
            return models.vit_b_16(weights=None)
        else:
            raise NotImplementedError(f"Base model {base_model} not supported") 

def load_checkpoint_rgb(self, ckpt_path, rgb_indices=[3,2,1]):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
    if 'conv1.weight' in state_dict:
        w = state_dict['conv1.weight']
        state_dict['conv1.weight'] = w[:, rgb_indices, :, :].contiguous()

    missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
    return missing, unexpected