import torch
import torch.nn as nn
from torchvision import models, transforms


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = backbone.avgpool
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224) — ImageNet-normalized
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)  # (B, 1280)


# Preprocessing transform — apply before passing frames to the model
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
