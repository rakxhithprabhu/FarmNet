"""Dataset discovery, splitting, and EfficientNet-B0 image preprocessing."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from config.settings import DISEASE_DATASET_DIR, DISEASE_IMAGE_SIZE, DISEASE_SEED

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}

train_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomContrast(0.1),
    ],
    name="disease_train_augmentation",
)


def _validate_dataset_root(dataset_root: str | Path = DISEASE_DATASET_DIR) -> Path:
    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Disease dataset directory not found: {root}")
    return root


def discover_class_names(dataset_root: str | Path = DISEASE_DATASET_DIR) -> list[str]:
    """Return non-empty class folders directly inside the configured root."""
    root = _validate_dataset_root(dataset_root)
    class_names = sorted(
        folder.name
        for folder in root.iterdir()
        if folder.is_dir()
        and any(path.suffix.lower() in IMAGE_EXTENSIONS for path in folder.iterdir())
    )
    if not class_names:
        raise ValueError(f"No image class folders found in {root}")
    return class_names


def dataset_info(dataset_root: str | Path = DISEASE_DATASET_DIR) -> tuple[Path, list[str], int]:
    """Return the resolved root, direct class folders, and direct image count."""
    root = _validate_dataset_root(dataset_root).resolve()
    class_names = discover_class_names(root)
    image_count = sum(
        1
        for class_name in class_names
        for path in (root / class_name).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return root, class_names, image_count


def split_dataset(
    dataset_root: str | Path = DISEASE_DATASET_DIR,
    seed: int = DISEASE_SEED,
) -> tuple[list[str], list[int], list[str], list[int], list[str], list[int]]:
    """Split every class into 70% train, 15% validation, and 15% test."""
    class_names = discover_class_names(dataset_root)
    rng = random.Random(seed)
    split_paths: list[list[str]] = [[], [], []]
    split_labels: list[list[int]] = [[], [], []]
    for label, class_name in enumerate(class_names):
        files = sorted(
            path
            for path in (Path(dataset_root) / class_name).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        rng.shuffle(files)
        train_end = int(len(files) * 0.70)
        validation_end = train_end + int(len(files) * 0.15)
        selections = (
            files[:train_end],
            files[train_end:validation_end],
            files[validation_end:],
        )
        for split_index, selected in enumerate(selections):
            split_paths[split_index].extend(str(path) for path in selected)
            split_labels[split_index].extend([label] * len(selected))
    return (
        split_paths[0], split_labels[0],
        split_paths[1], split_labels[1],
        split_paths[2], split_labels[2],
    )


def preprocess_image(image: tf.Tensor, training: bool = False) -> tf.Tensor:
    """Resize RGB data and apply the preprocessing expected by EfficientNet."""
    image = tf.image.resize(image, DISEASE_IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    if training:
        image = train_augmentation(image, training=True)
    return tf.keras.applications.efficientnet.preprocess_input(image)


def _load_path(path: tf.Tensor, label: tf.Tensor, training: bool) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.decode_image(
        tf.io.read_file(path), channels=3, expand_animations=False
    )
    return preprocess_image(image, training), label


def make_dataset(
    paths: list[str], labels: list[int], batch_size: int, training: bool
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, np.asarray(labels, dtype=np.int32))
    )
    if training:
        dataset = dataset.shuffle(
            max(len(paths), 1), seed=DISEASE_SEED, reshuffle_each_iteration=True
        )
    return dataset.map(
        lambda path, label: _load_path(path, label, training),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def create_datasets(
    dataset_root: str | Path = DISEASE_DATASET_DIR,
    batch_size: int = 32,
    seed: int = DISEASE_SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    class_names = discover_class_names(dataset_root)
    splits = split_dataset(dataset_root, seed)
    return (
        make_dataset(splits[0], splits[1], batch_size, True),
        make_dataset(splits[2], splits[3], batch_size, False),
        make_dataset(splits[4], splits[5], batch_size, False),
        class_names,
    )