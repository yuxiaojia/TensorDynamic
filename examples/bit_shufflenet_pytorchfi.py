#!/usr/bin/env python3
"""
ShuffleNet V2 x1.0 CIFAR-100 Evaluation with PyTorchFI Bit Flip Fault Injection (Position 30)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse

# Add fi module to path
_STANDALONE = Path(__file__).resolve().parent
if str(_STANDALONE) not in sys.path:
    sys.path.insert(0, str(_STANDALONE))

from fi.backends.pytorchfi_backend import setup_pytorchfi_bitflip_output_model, ResetEachForward


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_shufflenet_v2_cifar100(device):
    """Load ShuffleNet V2 x1.0 for CIFAR-100 from pytorch-cifar-models."""
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar100_shufflenetv2_x1_0",
        pretrained=True
    )
    model = model.to(device)
    model.eval()
    return model


def build_cifar100_test_loader(batch_size=100, num_workers=2):
    """Build CIFAR-100 test data loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761])
    ])

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


def evaluate_accuracy_with_fi(model, data_loader, device, target_layer, bit_count, bit_position=30, seed=42):
    """Evaluate top-1 accuracy with PyTorchFI bit flip fault injection."""
    model.eval()

    print(f"\nApplying PyTorchFI bit flip fault injection:")
    print(f"  Target layer: {target_layer}")
    print(f"  Bit count: {bit_count}")
    print(f"  Bit position: {bit_position}")
    print(f"  Seed: {seed}")

    # Setup fault injection
    batch_size = data_loader.batch_size
    input_shape = (3, 32, 32)  # CIFAR-100 image size

    pfi, fi_model = setup_pytorchfi_bitflip_output_model(
        model,
        batch_size=batch_size,
        input_shape=input_shape,
        target_layer_name=target_layer,
        bit_count=bit_count,
        bit_position=bit_position,
        seed=seed,
        device=device,
    )

    # Wrap with ResetEachForward to reset PyTorchFI cursor between batches
    fi_model = ResetEachForward(fi_model, pfi).to(device)
    fi_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = fi_model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description='ShuffleNet V2 x1.0 CIFAR-100 with PyTorchFI Bit Flip')
    parser.add_argument('--layer', type=str, default="conv1.0", help='Target layer for fault injection')
    parser.add_argument('--count', type=int, default=10, help='Number of bit flips to inject')
    args = parser.parse_args()

    print("ShuffleNet V2 x1.0 CIFAR-100 with PyTorchFI Bit Flip (Position 30)")
    print("=" * 60)

    # Setup
    batch_size = 100
    device = get_device()
    seed = 42
    bit_position = 30

    print(f"Device: {device}")
    print(f"Layer: {args.layer}")
    print(f"Bit count: {args.count}")
    print(f"Bit position: {bit_position}")

    # Load model
    model = load_shufflenet_v2_cifar100(device)

    # Load test data
    test_loader = build_cifar100_test_loader(batch_size=batch_size)

    # Evaluate with fault injection
    accuracy = evaluate_accuracy_with_fi(
        model, test_loader, device,
        target_layer=args.layer,
        bit_count=args.count,
        bit_position=bit_position,
        seed=seed
    )

    # Results
    print(f"\nTop-1 Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
