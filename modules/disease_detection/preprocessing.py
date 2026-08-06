"""
Image preprocessing and augmentation for the Disease Detection module.

Provides PyTorch-compatible transforms for training and inference.
Works with the PlantVillage dataset folder structure:
    dataset_root/
        class_a/
            img1.jpg
            ...
        class_b/
            ...
"""

from __future__ import annotations

from torchvision import transforms

from config.settings import DISEASE_IMAGE_SIZE

# ---------------------------------------------------------------------------
# Training transforms (with augmentation)
# ---------------------------------------------------------------------------
train_transforms = transforms.Compose(
    [
        transforms.Resize(DISEASE_IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# ---------------------------------------------------------------------------
# Validation / inference transforms (no augmentation)
# ---------------------------------------------------------------------------
val_transforms = transforms.Compose(
    [
        transforms.Resize(DISEASE_IMAGE_SIZE),
        transforms.CenterCrop(DISEASE_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)
