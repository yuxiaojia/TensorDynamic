#!/usr/bin/env python3
"""
ResNet-20 CIFAR-100 Evaluation (No Fault Injection)
Evaluates clean model accuracy on validation set.
"""

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent
_COMMON = _REPO_ROOT / "common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from fi.classification import build_model_factory, default_device, eval_top1_accuracy
from fi.datasets import build_cifar100_val_loader


def main() -> None:
    print("=" * 60)
    print("ResNet-20 CIFAR-100 Evaluation (No Fault Injection)")
    print("=" * 60)

    # Setup
    model_name = "resnet"
    batch_size = 100
    device = default_device()

    print(f"\nModel: ResNet-20 (cifar100_resnet20)")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")

    # Load model
    print("\nLoading model from torch hub...")
    model_factory = build_model_factory(model_name)
    model = model_factory(device)

    # Load validation data
    print("Loading CIFAR-100 validation data...")
    val_loader = build_cifar100_val_loader(batch_size=batch_size)

    # Evaluate
    print("\nEvaluating...")
    accuracy = eval_top1_accuracy(model, val_loader, device=device)

    # Results
    print("\n" + "=" * 60)
    print(f"Top-1 Accuracy: {accuracy:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
