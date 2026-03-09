from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn as nn
from torchvision import models as tv_models
from torchvision.models import GoogLeNet_Weights

from fi.utils import amp_autocast, seed_everything


@dataclass(frozen=True)
class ClassifierSpec:
    name: str
    dataset: str  # "cifar100" or "imagenet"
    input_shape: Sequence[int]


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cifar_model(model_name: str, *, device: torch.device) -> nn.Module:
    """
    Load CIFAR-100 pretrained models from chenyaofo/pytorch-cifar-models.
    """
    model_name = model_name.lower()
    if model_name == "resnet":
        hub_id = "cifar100_resnet20"
    elif model_name == "shufflenet":
        hub_id = "cifar100_shufflenetv2_x1_0"
    else:
        raise ValueError("CIFAR model must be one of: resnet, shufflenet")
    m = torch.hub.load("chenyaofo/pytorch-cifar-models", hub_id, pretrained=True)
    return m.to(device).eval()


def load_googlenet_imagenet(*, device: torch.device) -> nn.Module:
    weights = GoogLeNet_Weights.IMAGENET1K_V1
    # torchvision enforces aux_logits=True for pretrained weights
    # (aux branches are required to match the weight structure), but in eval()
    # the forward returns only the main logits tensor.
    m = tv_models.googlenet(weights=weights, aux_logits=True)
    return m.to(device).eval()


@torch.no_grad()
def eval_top1_accuracy(model: nn.Module, loader: Iterable, *, device: torch.device) -> float:
    total = 0
    correct = 0
    amp_ctx = amp_autocast(device)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with amp_ctx:
            logits = model(x)
        pred = logits.argmax(1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return 100.0 * correct / max(total, 1)


def classifier_spec(model_name: str) -> ClassifierSpec:
    model_name = model_name.lower()
    if model_name in {"resnet", "shufflenet"}:
        return ClassifierSpec(name=model_name, dataset="cifar100", input_shape=(3, 32, 32))
    if model_name == "googlenet":
        return ClassifierSpec(name=model_name, dataset="imagenet", input_shape=(3, 224, 224))
    raise ValueError("Unknown model. Use: resnet, shufflenet, googlenet")


def build_model_factory(model_name: str) -> Callable[[torch.device], nn.Module]:
    model_name = model_name.lower()
    if model_name in {"resnet", "shufflenet"}:
        return lambda device: load_cifar_model(model_name, device=device)
    if model_name == "googlenet":
        return lambda device: load_googlenet_imagenet(device=device)
    raise ValueError("Unknown model factory")


