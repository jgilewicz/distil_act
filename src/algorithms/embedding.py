import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights


class ImageEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 256, num_patches_side: int = 7, num_cameras: int = 2):
        super().__init__()
        model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.backbone = model.features

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((num_patches_side, num_patches_side))
        self.projection = nn.Linear(1536, embed_dim)

        num_patches = num_patches_side**2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.camera_embedding = nn.Embedding(num_cameras, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = images.shape

        images = images.view(B * N, C, H, W)
        features = self.backbone(images)
        features = self.pool(features)
        features = features.flatten(2).permute(0, 2, 1)
        features = self.projection(features)

        features = features.view(B, N, -1, features.shape[-1])

        cam_ids = torch.arange(N, device=images.device)
        cam_tokens = self.camera_embedding(cam_ids)
        features = features + cam_tokens.unsqueeze(0).unsqueeze(2)
        features = features + self.pos_embedding.unsqueeze(1)

        features = features.view(B, N * features.shape[2], features.shape[-1])
        return features
