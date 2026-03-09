#!/usr/bin/env python3
"""
MobileNetV2 CIFAR-100 Evaluation (No Fault Injection)
Uses pytorch-cifar-models: https://github.com/chenyaofo/pytorch-cifar-models
Standalone script - evaluates MobileNetV2 x1.0 on CIFAR-100.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_mobilenet_v2_cifar100(device):
    """Load MobileNetV2 x1.0 for CIFAR-100 from pytorch-cifar-models."""
    # Load pretrained model from pytorch-cifar-models
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar100_mobilenetv2_x1_0",
        pretrained=True
    )
    model = model.to(device)
    model.eval()
    return model


def build_cifar100_test_loader(batch_size=100, num_workers=2):
    """Build CIFAR-100 test data loader."""
    # CIFAR-100 normalization (same as CIFAR-10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761])
    ])

    # Load CIFAR-100 test set
    test_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return test_loader


def evaluate_accuracy(model, data_loader, device):
    """Evaluate top-1 accuracy on the test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def main() -> None:
    print("MobileNetV2 x1.0 CIFAR-100 Evaluation")
    print("=" * 60)

    # Setup
    batch_size = 100
    device = get_device()

    print(f"Device: {device}")

    # Load model
    model = load_mobilenet_v2_cifar100(device)

    # Load test data
    test_loader = build_cifar100_test_loader(batch_size=batch_size)

    # Evaluate
    accuracy = evaluate_accuracy(model, test_loader, device=device)

    # Results
    print(f"Top-1 Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
