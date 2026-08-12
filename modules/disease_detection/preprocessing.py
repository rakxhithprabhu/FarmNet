"""Disease-specific preprocessing, tabular encoding, and dataset helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

try:  # pragma: no cover - optional dependency
    import albumentations as A
except ImportError:  # pragma: no cover
    A = None

from config.settings import (
    DISEASE_DATASET_PATH,
    DISEASE_IMAGE_DIR,
    DISEASE_IMAGE_SIZE,
    DISEASE_SEASON_COLUMN,
    DISEASE_TABULAR_FEATURES,
)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225] 

if A is not None:
    alb_train_transform = A.Compose(
        [
            A.Resize(height=DISEASE_IMAGE_SIZE[0], width=DISEASE_IMAGE_SIZE[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=MEAN, std=STD),
        ]
    )
    alb_val_transform = A.Compose(
        [
            A.Resize(height=DISEASE_IMAGE_SIZE[0], width=DISEASE_IMAGE_SIZE[1]),
            A.Normalize(mean=MEAN, std=STD),
        ]
    )
else:
    alb_train_transform = None
    alb_val_transform = None

train_transforms = transforms.Compose(
    [
        transforms.Resize(DISEASE_IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(DISEASE_IMAGE_SIZE),
        transforms.CenterCrop(DISEASE_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)


def get_image_transform(split: str) -> Any:
    """Return the correct augmentation pipeline for a training or evaluation split."""
    if split == "train":
        return alb_train_transform or train_transforms
    if split in {"val", "test"}:
        return alb_val_transform or val_transforms
    raise ValueError(f"Unsupported split: {split}")


class DiseaseTabularPreprocessor:
    """Fit tabular preprocessing on the training split and reuse it for eval."""

    def __init__(self) -> None:
        self.numeric_columns = [
            "temperature",
            "humidity",
            "rainfall",
            "soil_moisture",
            "pH",
            "N",
            "P",
            "K",
        ]
        self.season_categories: list[str] = []
        self.fill_values: dict[str, float] = {}
        self.numeric_means: dict[str, float] = {}
        self.numeric_stds: dict[str, float] = {}
        self.output_dim = 0

    def _ensure_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        work = frame.copy()
        for column in self.numeric_columns:
            if column not in work.columns:
                work[column] = np.nan
        if DISEASE_SEASON_COLUMN not in work.columns:
            work[DISEASE_SEASON_COLUMN] = "Unknown"
        work[DISEASE_SEASON_COLUMN] = work[DISEASE_SEASON_COLUMN].fillna("Unknown")
        work[DISEASE_SEASON_COLUMN] = work[DISEASE_SEASON_COLUMN].astype(str)
        work[DISEASE_SEASON_COLUMN] = work[DISEASE_SEASON_COLUMN].str.title()
        return work

    def fit(self, frame: pd.DataFrame) -> "DiseaseTabularPreprocessor":
        work = self._ensure_columns(frame)
        for column in self.numeric_columns:
            values = pd.to_numeric(work[column], errors="coerce")
            self.fill_values[column] = float(values.median())
            self.numeric_means[column] = float(values.mean())
            self.numeric_stds[column] = float(values.std(ddof=0) or 1.0)
        self.season_categories = sorted(
            {value for value in work[DISEASE_SEASON_COLUMN].astype(str).tolist() if value}
        )
        self.output_dim = len(self.numeric_columns) + len(self.season_categories)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        work = self._ensure_columns(frame)
        rows: list[list[float]] = []
        for _, row in work.iterrows():
            numeric_values: list[float] = []
            for column in self.numeric_columns:
                value = pd.to_numeric(row[column], errors="coerce")
                if pd.isna(value):
                    value = self.fill_values.get(column, 0.0)
                value = float(value)
                if column in self.numeric_means and column in self.numeric_stds:
                    value = (value - self.numeric_means[column]) / self.numeric_stds[column]
                numeric_values.append(value)
            season_vector = [1.0 if row[DISEASE_SEASON_COLUMN] == season else 0.0 for season in self.season_categories]
            rows.append(numeric_values + season_vector)
        return np.asarray(rows, dtype=np.float32)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.fit(frame)
        return self.transform(frame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric_columns": self.numeric_columns,
            "season_categories": self.season_categories,
            "fill_values": self.fill_values,
            "numeric_means": self.numeric_means,
            "numeric_stds": self.numeric_stds,
            "output_dim": self.output_dim,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiseaseTabularPreprocessor":
        preprocessor = cls()
        preprocessor.numeric_columns = list(payload.get("numeric_columns", cls().numeric_columns))
        preprocessor.season_categories = list(payload.get("season_categories", []))
        preprocessor.fill_values = {str(k): float(v) for k, v in payload.get("fill_values", {}).items()}
        preprocessor.numeric_means = {str(k): float(v) for k, v in payload.get("numeric_means", {}).items()}
        preprocessor.numeric_stds = {str(k): float(v) for k, v in payload.get("numeric_stds", {}).items()}
        preprocessor.output_dim = int(payload.get("output_dim", 0))
        return preprocessor


class DiseaseDataset(Dataset):
    """Image + tabular dataset for multimodal disease training."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str | Path | None = None,
        transform: Any | None = None,
        tabular_features: np.ndarray | None = None,
        label_map: dict[str, dict[str, int]] | None = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir or DISEASE_IMAGE_DIR)
        self.transform = transform or val_transforms
        self.tabular_features = tabular_features
        self.label_map = label_map or {}

    def __len__(self) -> int:
        return len(self.dataframe)

    def _resolve_image_path(self, image_name: str) -> Path:
        candidate_paths = [
            Path(image_name),
            self.image_dir / image_name,
            self.image_dir / image_name.split("/")[-1],
            Path(image_name).name,
        ]
        for path in candidate_paths:
            if path.exists():
                return path
        raise FileNotFoundError(f"Could not find image: {image_name}")

    def _load_image(self, image_name: str) -> torch.Tensor:
        image = Image.open(self._resolve_image_path(image_name)).convert("RGB")
        transform = self.transform
        if transform is None:
            image_tensor = val_transforms(image)
        elif A is not None and hasattr(transform, "__class__") and transform.__class__.__module__.startswith("albumentations"):
            image_np = np.array(image)
            transformed = transform(image=image_np)
            image_array = transformed["image"] if isinstance(transformed, dict) else transformed
            image_tensor = torch.from_numpy(np.array(image_array)).permute(2, 0, 1).float()
        else:
            image_tensor = transform(image)
            if not isinstance(image_tensor, torch.Tensor):
                image_tensor = torch.from_numpy(np.array(image_tensor))
            if image_tensor.ndim == 3 and image_tensor.shape[0] != 3:
                image_tensor = image_tensor.permute(2, 0, 1)
        return image_tensor.float()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        image_tensor = self._load_image(str(row["image_name"]))
        tabular = torch.from_numpy(self.tabular_features[index]).float() if self.tabular_features is not None else torch.zeros(1)

        crop_idx = self.label_map.get("crop", {}).get(str(row["crop"]), 0)
        disease_idx = self.label_map.get("disease", {}).get(str(row["disease"]), 0)
        severity_idx = self.label_map.get("severity", {}).get(str(row["severity"]), 0)

        return (
            image_tensor,
            tabular,
            torch.tensor(crop_idx, dtype=torch.long),
            torch.tensor(disease_idx, dtype=torch.long),
            torch.tensor(severity_idx, dtype=torch.long),
        )


def load_disease_data(csv_path: str | None = None) -> pd.DataFrame:
    """Load disease labels from the existing data directory if present."""
    candidates: list[Path] = []
    if csv_path:
        candidates.append(Path(csv_path))
    candidates.extend([DISEASE_DATASET_PATH, Path("data/disease/labels.csv")])

    for candidate in candidates:
        if candidate.exists():
            df = pd.read_csv(candidate)
            break
    else:
        raise FileNotFoundError(
            "No disease labels file was found. Expected a CSV at data/disease/labels.csv "
            "or pass the path explicitly."
        )

    required_columns = {
        "image_name",
        "crop",
        "disease",
        "severity",
        "temperature",
        "humidity",
        "rainfall",
        "season",
        "soil_moisture",
        "pH",
        "N",
        "P",
        "K",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Disease labels file is missing required columns: {sorted(missing)}")

    df = df.copy()
    for column in ["crop", "disease", "severity", "season"]:
        if column in df.columns:
            df[column] = df[column].fillna("Unknown").astype(str)
    for column in ["temperature", "humidity", "rainfall", "soil_moisture", "pH", "N", "P", "K"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def build_label_maps(dataframe: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Create deterministic integer mappings for crop, disease, and severity."""
    return {
        "crop": {label: idx for idx, label in enumerate(sorted(dataframe["crop"].astype(str).unique().tolist()))},
        "disease": {label: idx for idx, label in enumerate(sorted(dataframe["disease"].astype(str).unique().tolist()))},
        "severity": {label: idx for idx, label in enumerate(sorted(dataframe["severity"].astype(str).unique().tolist()))},
    }


def save_preprocessor(preprocessor: DiseaseTabularPreprocessor, path: str | Path | None = None) -> None:
    """Persist the fitted tabular preprocessor metadata to disk."""
    output_path = Path(path or DISEASE_DATASET_PATH.parent / "preprocessor.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(preprocessor.to_dict(), handle, indent=2)


def load_preprocessor(path: str | Path | None = None) -> DiseaseTabularPreprocessor:
    """Load the persisted tabular preprocessor metadata."""
    input_path = Path(path or DISEASE_DATASET_PATH.parent / "preprocessor.json")
    if not input_path.exists():
        raise FileNotFoundError(f"No preprocessor metadata found at {input_path}")
    with input_path.open("r", encoding="utf-8") as handle:
        return DiseaseTabularPreprocessor.from_dict(json.load(handle))
