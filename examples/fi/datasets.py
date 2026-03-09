from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torchvision import datasets, transforms

from fi.paths import repo_path


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_cifar100_val_loader(
    *,
    batch_size: int,
    num_workers: int = 4,
    root: Path | None = None,
) -> torch.utils.data.DataLoader:
    """
    CIFAR-100 test split, with normalization.
    Downloads to <repo>/.data by default.
    """
    if root is None:
        root = repo_path(".data")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )

    dataset = datasets.CIFAR100(
        root=str(root),
        train=False,
        download=True,
        transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def ensure_imagenet_root(root: Path) -> None:
    """
    We expect:
      <root>/val/<wnid>/<image>.JPEG

    Note: we intentionally do NOT require the torchvision ImageNet devkit archives.
    """
    if not root.exists():
        raise FileNotFoundError(f"ImageNet root not found: {root}")
    if not (root / "val").exists():
        raise FileNotFoundError(
            f"ImageNet val split not found: {(root / 'val')}. "
            "Expected ImageNet layout: <root>/val/<wnid>/*.JPEG"
        )


def build_imagenet_val_loader(
    *,
    imagenet_root: Path,
    batch_size: int,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    """ImageNet val split loader (directory-based, no devkit required)."""
    ensure_imagenet_root(imagenet_root)

    transform = transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    # Use ImageFolder to avoid torchvision.datasets.ImageNet devkit/archive requirements.
    dataset = datasets.ImageFolder(
        root=str(imagenet_root / "val"),
        transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


