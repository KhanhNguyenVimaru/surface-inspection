from pathlib import Path
from typing import Dict, Tuple

from torchvision import datasets, transforms

from src.config import IMG_SIZE


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def load_imagefolder_splits(processed_dir: str):
    train_tf, eval_tf = build_transforms()
    processed_path = Path(processed_dir)

    train_ds = datasets.ImageFolder(processed_path / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(processed_path / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(processed_path / "test", transform=eval_tf)
    return train_ds, val_ds, test_ds


def class_distribution(imagefolder_ds) -> Dict[str, int]:
    counts: Dict[str, int] = {cls_name: 0 for cls_name in imagefolder_ds.classes}
    for _, label in imagefolder_ds.samples:
        cls_name = imagefolder_ds.classes[label]
        counts[cls_name] += 1
    return counts
