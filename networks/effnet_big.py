"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

# model_dict = {
#     'efficientnet-b0': [efficientnet-b0, 1280],
#     'resnet34': [resnet34, 512],
#     'resnet50': [resnet50, 2048],
#     'resnet101': [resnet101, 2048],
# }

class SupConEffNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, head='mlp', feat_dim=128):
        super(SupConEffNet, self).__init__()
        # _, dim_in = model_dict[name]
        dim_in = 1280
        self.encoder = EfficientNet.from_name('efficientnet-b0', num_classes=2, include_top=False)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        print(feat.shape)
        feat = F.normalize(self.head(feat), dim=1)
        print(feat.shape)
        return feat


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes=4):
        super(LinearClassifier, self).__init__()
        feat_dim = 1280
        self.fc1 = nn.Linear(feat_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, features):
        features = features.view(features.shape[0], -1)
        x = self.fc1(features)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
