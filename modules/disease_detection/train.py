"""Training pipeline for the multimodal crop disease detection model."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from models.disease_detection_model import MultiTaskDiseaseModel

try:  # pragma: no cover
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Any, **_: Any) -> Any:  # type: ignore[override]
        return iterable

from config.settings import (
    DISEASE_BATCH_SIZE,
    DISEASE_CLASS_NAMES_PATH,
    DISEASE_CROP_LOSS_WEIGHT,
    DISEASE_DISEASE_LOSS_WEIGHT,
    DISEASE_EARLY_STOPPING_PATIENCE,
    DISEASE_EPOCHS,
    DISEASE_LEARNING_RATE,
    DISEASE_MODEL_PATH,
    DISEASE_DATASET_PATH,
    DISEASE_METADATA_PATH,
    DISEASE_NUM_WORKERS,
    DISEASE_RANDOM_SEED,
    DISEASE_SEVERITY_LOSS_WEIGHT,
    DISEASE_TABULAR_FEATURES,
    DISEASE_TEST_SPLIT,
    DISEASE_VALIDATION_SPLIT,
)
from modules.disease_detection.preprocessing import (
    DiseaseDataset,
    DiseaseTabularPreprocessor,
    build_label_maps,
    get_image_transform,
    load_disease_data,
    save_preprocessor,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int = DISEASE_RANDOM_SEED) -> None:
    """Make training deterministic across CPU/GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _CompatSingleTaskModel(nn.Module):
    """Compatibility wrapper used by existing tests and simple inference."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = MultiTaskDiseaseModel(
            num_crop_classes=num_classes,
            num_disease_classes=num_classes,
            num_severity_classes=num_classes,
            tabular_input_dim=10,
        )
        self.num_classes = num_classes

    def forward(self, images: torch.Tensor, tabular: torch.Tensor | None = None) -> torch.Tensor:
        if tabular is None:
            tabular = torch.zeros((images.size(0), 10), device=images.device, dtype=images.dtype)
        outputs = self.backbone(images, tabular)
        return outputs["disease"]


def _build_model(num_classes: int) -> nn.Module:
    """Return a classifier compatible with the existing disease tests."""
    return _CompatSingleTaskModel(num_classes)


def _get_task_names(dataframe: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    crop_names = sorted(dataframe["crop"].astype(str).unique().tolist())
    disease_names = sorted(dataframe["disease"].astype(str).unique().tolist())
    severity_names = sorted(dataframe["severity"].astype(str).unique().tolist())
    return crop_names, disease_names, severity_names


def _evaluate_outputs(
    predictions: np.ndarray,
    targets: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(targets, predictions, average="macro", zero_division=0)),
        "f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(len(labels)))).tolist(),
        "labels": labels,
    }


def train(
    dataset_path: str | None = None,
    epochs: int = DISEASE_EPOCHS,
    batch_size: int = DISEASE_BATCH_SIZE,
    lr: float = DISEASE_LEARNING_RATE,
    save: bool = True,
) -> dict[str, Any]:
    """Train the multimodal disease model and persist the best checkpoint."""
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    dataframe = load_disease_data(dataset_path)
    if dataframe.empty:
        raise ValueError("Disease labels file is empty.")

    crop_names, disease_names, severity_names = _get_task_names(dataframe)
    label_maps = build_label_maps(dataframe)

    feature_columns = [col for col in DISEASE_TABULAR_FEATURES if col in dataframe.columns]
    train_df, val_df, test_df = train_test_split(dataframe, feature_columns, label_maps)

    preprocessor = DiseaseTabularPreprocessor().fit(train_df[feature_columns + ["season"]])
    train_tabular = preprocessor.transform(train_df[feature_columns + ["season"]])
    val_tabular = preprocessor.transform(val_df[feature_columns + ["season"]])
    test_tabular = preprocessor.transform(test_df[feature_columns + ["season"]])

    train_dataset = DiseaseDataset(
        train_df,
        transform=get_image_transform("train"),
        tabular_features=train_tabular,
        label_map=label_maps,
    )
    val_dataset = DiseaseDataset(
        val_df,
        transform=get_image_transform("val"),
        tabular_features=val_tabular,
        label_map=label_maps,
    )
    test_dataset = DiseaseDataset(
        test_df,
        transform=get_image_transform("test"),
        tabular_features=test_tabular,
        label_map=label_maps,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=DISEASE_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DISEASE_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DISEASE_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = MultiTaskDiseaseModel(
        num_crop_classes=len(crop_names),
        num_disease_classes=len(disease_names),
        num_severity_classes=len(severity_names),
        tabular_input_dim=preprocessor.output_dim,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1, factor=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    history: list[tuple[float, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, tabular, crop_labels, disease_labels, severity_labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            images = images.to(device)
            tabular = tabular.to(device)
            crop_labels = crop_labels.to(device)
            disease_labels = disease_labels.to(device)
            severity_labels = severity_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(images, tabular)
                crop_loss = criterion(outputs["crop"], crop_labels)
                disease_loss = criterion(outputs["disease"], disease_labels)
                severity_loss = criterion(outputs["severity"], severity_labels)
                loss = (
                    DISEASE_CROP_LOSS_WEIGHT * crop_loss
                    + DISEASE_DISEASE_LOSS_WEIGHT * disease_loss
                    + DISEASE_SEVERITY_LOSS_WEIGHT * severity_loss
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * images.size(0)

        train_loss = running_loss / len(train_dataset)
        logger.info("Epoch %d/%d - train_loss=%.4f", epoch, epochs, train_loss)

        model.eval()
        val_running_loss = 0.0
        predictions: dict[str, list[int]] = {"crop": [], "disease": [], "severity": []}
        targets: dict[str, list[int]] = {"crop": [], "disease": [], "severity": []}
        with torch.no_grad():
            for images, tabular, crop_labels, disease_labels, severity_labels in val_loader:
                images = images.to(device)
                tabular = tabular.to(device)
                outputs = model(images, tabular)
                crop_loss = criterion(outputs["crop"], crop_labels.to(device))
                disease_loss = criterion(outputs["disease"], disease_labels.to(device))
                severity_loss = criterion(outputs["severity"], severity_labels.to(device))
                loss = (
                    DISEASE_CROP_LOSS_WEIGHT * crop_loss
                    + DISEASE_DISEASE_LOSS_WEIGHT * disease_loss
                    + DISEASE_SEVERITY_LOSS_WEIGHT * severity_loss
                )
                val_running_loss += float(loss.item()) * images.size(0)
                predictions["crop"].extend(outputs["crop"].argmax(dim=1).cpu().tolist())
                predictions["disease"].extend(outputs["disease"].argmax(dim=1).cpu().tolist())
                predictions["severity"].extend(outputs["severity"].argmax(dim=1).cpu().tolist())
                targets["crop"].extend(crop_labels.tolist())
                targets["disease"].extend(disease_labels.tolist())
                targets["severity"].extend(severity_labels.tolist())

        val_loss = val_running_loss / len(val_dataset)
        history.append((train_loss, val_loss))
        logger.info("Epoch %d/%d - val_loss=%.4f", epoch, epochs, val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= DISEASE_EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save:
        DISEASE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), DISEASE_MODEL_PATH)
        save_preprocessor(preprocessor, DISEASE_DATASET_PATH.parent / "preprocessor.json")
        with DISEASE_CLASS_NAMES_PATH.open("w", encoding="utf-8") as handle:
            json.dump(disease_names, handle, indent=2)
        metadata = {
            "crop_names": crop_names,
            "disease_names": disease_names,
            "severity_names": severity_names,
            "feature_columns": feature_columns + ["season"],
            "label_maps": label_maps,
            "preprocessor": preprocessor.to_dict(),
            "loss_weights": {
                "crop": DISEASE_CROP_LOSS_WEIGHT,
                "disease": DISEASE_DISEASE_LOSS_WEIGHT,
                "severity": DISEASE_SEVERITY_LOSS_WEIGHT,
            },
            "split_config": {
                "validation_split": DISEASE_VALIDATION_SPLIT,
                "test_split": DISEASE_TEST_SPLIT,
            },
        }
        with DISEASE_METADATA_PATH.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        logger.info("Saved disease model to %s", DISEASE_MODEL_PATH)

    return {
        "num_crop_classes": len(crop_names),
        "num_disease_classes": len(disease_names),
        "num_severity_classes": len(severity_names),
        "history": history,
        "best_val_loss": float(best_val_loss),
        "metrics": {
            "crop": _evaluate_outputs(np.asarray(predictions["crop"]), np.asarray(targets["crop"]), crop_names),
            "disease": _evaluate_outputs(np.asarray(predictions["disease"]), np.asarray(targets["disease"]), disease_names),
            "severity": _evaluate_outputs(np.asarray(predictions["severity"]), np.asarray(targets["severity"]), severity_names),
        },
    }


def train_test_split(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    label_maps: dict[str, dict[str, int]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train, validation, and test subsets while preserving class balance where possible."""
    from sklearn.model_selection import train_test_split as sklearn_split

    if len(dataframe) < 6:
        train_df = dataframe.iloc[: max(1, len(dataframe) - 2)].copy()
        val_df = dataframe.iloc[max(1, len(dataframe) - 2): max(2, len(dataframe) - 1)].copy()
        test_df = dataframe.iloc[max(2, len(dataframe) - 1):].copy()
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    total_holdout_size = DISEASE_VALIDATION_SPLIT + DISEASE_TEST_SPLIT
    if total_holdout_size >= 1.0:
        raise ValueError("Validation and test splits must sum to less than 1.0")

    try:
        train_df, temp_df = sklearn_split(
            dataframe,
            test_size=total_holdout_size,
            random_state=DISEASE_RANDOM_SEED,
            stratify=dataframe["disease"],
        )
    except ValueError:
        train_df, temp_df = sklearn_split(
            dataframe,
            test_size=total_holdout_size,
            random_state=DISEASE_RANDOM_SEED,
        )

    val_fraction = DISEASE_VALIDATION_SPLIT / total_holdout_size
    try:
        val_df, test_df = sklearn_split(
            temp_df,
            test_size=1.0 - val_fraction,
            random_state=DISEASE_RANDOM_SEED,
            stratify=temp_df["disease"],
        )
    except ValueError:
        val_df, test_df = sklearn_split(
            temp_df,
            test_size=1.0 - val_fraction,
            random_state=DISEASE_RANDOM_SEED,
        )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        summary = train()
    except FileNotFoundError as exc:
        logger.error("Training failed: %s", exc)
        raise
    print(json.dumps(summary, indent=2))
