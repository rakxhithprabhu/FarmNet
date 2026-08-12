"""Multimodal disease-detection model architecture shared by training and inference."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class MultiTaskDiseaseModel(nn.Module):
    """EfficientNet-B0 image encoder + tabular MLP with three task heads."""

    def __init__(
        self,
        num_crop_classes: int,
        num_disease_classes: int,
        num_severity_classes: int,
        tabular_input_dim: int,
    ) -> None:
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        backbone.classifier = nn.Identity()
        self.image_backbone = backbone
        self.image_embedding_dim = 1280
        self.image_dropout = nn.Dropout(0.2)

        self.tabular_projection = nn.Sequential(
            nn.Linear(tabular_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
        )
        self.shared_layers = nn.Sequential(
            nn.Linear(self.image_embedding_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )
        self.crop_head = nn.Linear(128, num_crop_classes)
        self.disease_head = nn.Linear(128, num_disease_classes)
        self.severity_head = nn.Linear(128, num_severity_classes)

    def _image_embedding(self, images: torch.Tensor) -> torch.Tensor:
        features = self.image_backbone.features(images)
        features = self.image_backbone.avgpool(features)
        features = torch.flatten(features, 1)
        return self.image_dropout(features)

    def forward(
        self,
        images: torch.Tensor,
        tabular: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        image_embedding = self._image_embedding(images)
        if tabular is None:
            tabular = torch.zeros((images.size(0), 1), device=images.device, dtype=images.dtype)
        tabular_embedding = self.tabular_projection(tabular)
        fused = torch.cat([image_embedding, tabular_embedding], dim=1)
        shared = self.shared_layers(fused)
        return {
            "crop": self.crop_head(shared),
            "disease": self.disease_head(shared),
            "severity": self.severity_head(shared),
        }
