"""
Training script for the Crop Disease Detection module.

Uses transfer learning with ResNet-18 (lightweight, fast to train).
Expects the PlantVillage dataset laid out as:

    <dataset_root>/
        Tomato___Late_blight/
            img001.jpg
            ...
        Potato___Early_blight/
            ...

The script:
1. Applies augmentation / preprocessing
2. Splits into train / val sets
3. Fine-tunes a pre-trained ResNet-18
4. Saves the model (.pt) and class-name mapping
5. Prints precision, recall, F1, and confusion matrix
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models
from sklearn.metrics import classification_report, confusion_matrix

from smart_agriculture.config.settings import (
    DISEASE_BATCH_SIZE,
    DISEASE_CLASS_NAMES_PATH,
    DISEASE_EPOCHS,
    DISEASE_IMAGE_SIZE,
    DISEASE_LEARNING_RATE,
    DISEASE_MODEL_PATH,
    DISEASE_VALIDATION_SPLIT,
)
from smart_agriculture.modules.disease_detection.preprocessing import (
    train_transforms,
    val_transforms,
)

logger = logging.getLogger(__name__)


def _build_model(num_classes: int) -> nn.Module:
    """Return a ResNet-18 with its final FC layer replaced."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze all layers except the final classifier
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(
    dataset_root: str,
    epochs: int = DISEASE_EPOCHS,
    batch_size: int = DISEASE_BATCH_SIZE,
    lr: float = DISEASE_LEARNING_RATE,
    save: bool = True,
) -> dict:
    """
    Fine-tune ResNet-18 on the PlantVillage dataset.

    Parameters
    ----------
    dataset_root : str
        Path to the root folder containing one sub-folder per class.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for DataLoader.
    lr : float
        Learning rate for Adam optimiser.
    save : bool
        Whether to persist the model and class names.

    Returns
    -------
    dict
        Training summary including final metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ---- dataset & split ----
    full_dataset = datasets.ImageFolder(dataset_root, transform=train_transforms)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    val_size = int(len(full_dataset) * DISEASE_VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    # Override transforms for validation subset
    val_ds.dataset = datasets.ImageFolder(dataset_root, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ---- model ----
    model = _build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    # ---- training loop ----
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / train_size
        logger.info("Epoch %d/%d  loss=%.4f", epoch, epochs, epoch_loss)

    # ---- evaluation ----
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()
    logger.info("Classification report:\n%s",
                classification_report(all_labels, all_preds, target_names=class_names))

    # ---- save ----
    if save:
        DISEASE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), DISEASE_MODEL_PATH)
        with open(DISEASE_CLASS_NAMES_PATH, "w") as f:
            json.dump(class_names, f)
        logger.info("Model saved to %s", DISEASE_MODEL_PATH)

    return {
        "num_classes": num_classes,
        "class_names": class_names,
        "classification_report": report,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m smart_agriculture.modules.disease_detection.train <dataset_root>")
        sys.exit(1)
    summary = train(sys.argv[1])
    print(json.dumps({k: v for k, v in summary.items() if k != "confusion_matrix"}, indent=2))
