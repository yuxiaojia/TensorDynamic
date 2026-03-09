from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from pytorchfi.core import fault_injection

from fi.utils import seed_everything

_LAST_CORRUPTED: List[Tuple[int, float, float]] = []


def reset_last_corrupted() -> None:
    global _LAST_CORRUPTED
    _LAST_CORRUPTED = []


def get_last_corrupted() -> List[Tuple[int, float, float]]:
    return _LAST_CORRUPTED


def list_leaf_scopes_conv_linear(model: nn.Module) -> List[str]:
    """Leaf Conv/Linear list for sweeps."""
    names: List[str] = []
    for name, m in model.named_modules():
        if len(list(m.children())) == 0 and isinstance(
            m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
        ):
            names.append(name)
    return names


def _iter_leaf_modules_children_order(
    model: nn.Module, prefix: str = ""
) -> List[Tuple[str, nn.Module]]:
    out: List[Tuple[str, nn.Module]] = []
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if len(list(child.children())) == 0:
            out.append((full_name, child))
        else:
            out.extend(_iter_leaf_modules_children_order(child, full_name))
    return out


LayerTypes = Union[Sequence[type], Sequence[Union[type, str]]]


def build_pytorchfi_layer_name_to_index(
    model: nn.Module, layer_types: LayerTypes
) -> Dict[str, int]:
    name_to_idx: Dict[str, int] = {}
    idx = 0
    leafs = _iter_leaf_modules_children_order(model)

    match_all = any(isinstance(t, str) and t == "all" for t in layer_types)
    for full_name, m in leafs:
        if match_all:
            name_to_idx[full_name] = idx
            idx += 1
            continue
        for t in layer_types:
            if isinstance(t, type) and isinstance(m, t):
                name_to_idx[full_name] = idx
                idx += 1
                break
    return name_to_idx


def setup_pytorchfi(
    model: nn.Module,
    *,
    batch_size: int,
    input_shape: Sequence[int],
    layer_types: Sequence[type],
    use_cuda: Optional[bool] = None,
) -> fault_injection:
    if use_cuda is None:
        use_cuda = next(model.parameters()).is_cuda
    return fault_injection(
        model=model,
        batch_size=int(batch_size),
        input_shape=list(input_shape),
        layer_types=list(layer_types),
        use_cuda=bool(use_cuda),
    )


def make_extreme_output_hook(
    pfi: fault_injection,
    *,
    target_layer_idx: int,
    extreme_count: int,
    extreme_factor: float,
) -> callable:
    """
    Forward hook mirroring ErrorInjector(mode='extreme') but only at target_layer_idx.
    """

    def hook(module, input_val, output):
        try:
            cur = pfi.get_current_layer()
            if cur != target_layer_idx:
                return
            if not torch.is_tensor(output) or output.numel() == 0 or not output.is_floating_point():
                return

            numel = int(output.numel())
            actual = min(int(extreme_count), numel)
            if actual <= 0:
                return

            flat_indices = torch.randperm(numel, device=output.device)[:actual]
            flat = output.reshape(-1)
            signs = (
                torch.randint(0, 2, (actual,), device=output.device, dtype=torch.int8) * 2 - 1
            ).to(dtype=flat.dtype)

            global _LAST_CORRUPTED
            _LAST_CORRUPTED = []

            for i in range(actual):
                idx = int(flat_indices[i].item())
                orig = flat[idx].clone()
                corr = orig + (orig * float(extreme_factor) * signs[i])
                flat[idx] = corr
                _LAST_CORRUPTED.append(
                    (idx, float(orig.detach().cpu().item()), float(corr.detach().cpu().item()))
                )
        finally:
            # Keep PyTorchFI cursor in sync.
            pfi.updateLayer()

    return hook


def make_bitflip_output_hook(
    pfi: fault_injection,
    *,
    target_layer_idx: int,
    bit_count: int,
    bit_position: Optional[int] = None,
) -> callable:
    """
    Forward hook for bit flip fault injection at target_layer_idx.

    Args:
        pfi: PyTorchFI fault_injection instance
        target_layer_idx: Layer index to inject faults
        bit_count: Number of values to flip bits in
        bit_position: Specific bit position to flip (0-31 for float32), or None for random
    """

    def hook(module, input_val, output):
        try:
            cur = pfi.get_current_layer()
            if cur != target_layer_idx:
                return
            if not torch.is_tensor(output) or output.numel() == 0 or not output.is_floating_point():
                return

            numel = int(output.numel())
            actual = min(int(bit_count), numel)
            if actual <= 0:
                return

            flat_indices = torch.randperm(numel, device=output.device)[:actual]
            flat = output.reshape(-1)

            global _LAST_CORRUPTED
            _LAST_CORRUPTED = []

            for i in range(actual):
                idx = int(flat_indices[i].item())
                orig = flat[idx].clone()

                # Convert to int32 representation for bit manipulation
                orig_int = orig.view(torch.int32)

                # Determine bit position to flip
                if bit_position is not None:
                    bit_pos = bit_position % 32
                else:
                    bit_pos = torch.randint(0, 32, (1,), device=output.device).item()

                # Flip the bit
                mask = 1 << bit_pos
                flipped_int = orig_int ^ mask

                # Convert back to float
                corr = flipped_int.view(torch.float32)
                flat[idx] = corr

                _LAST_CORRUPTED.append(
                    (idx, float(orig.detach().cpu().item()), float(corr.detach().cpu().item()))
                )
        finally:
            # Keep PyTorchFI cursor in sync.
            pfi.updateLayer()

    return hook


def setup_pytorchfi_extreme_output_model(
    model: nn.Module,
    *,
    batch_size: int,
    input_shape: Sequence[int],
    target_layer_name: str,
    extreme_count: int,
    extreme_factor: float,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[fault_injection, nn.Module]:
    if seed is not None:
        seed_everything(int(seed))

    if device is not None:
        model = model.to(device)

    layer_types = [nn.Conv2d, nn.Linear]
    pfi = setup_pytorchfi(
        model,
        batch_size=batch_size,
        input_shape=input_shape,
        layer_types=layer_types,
        use_cuda=(device is not None and device.type == "cuda") if device is not None else None,
    )

    name_to_idx = build_pytorchfi_layer_name_to_index(model, layer_types)
    if target_layer_name not in name_to_idx:
        raise KeyError(
            f"target_layer_name='{target_layer_name}' not found in PyTorchFI layer map. "
            f"Example names: {list(name_to_idx.keys())[:10]}"
        )
    target_idx = int(name_to_idx[target_layer_name])

    # Provide one valid injection coordinate to satisfy bounds checks.
    layer_dim = int(pfi.get_layer_dim(target_idx))
    dim1 = [0]
    if layer_dim <= 2:
        dim2 = [None]
        dim3 = [None]
    elif layer_dim == 3:
        dim2 = [0]
        dim3 = [None]
    else:
        dim2 = [0]
        dim3 = [0]

    hook = make_extreme_output_hook(
        pfi,
        target_layer_idx=target_idx,
        extreme_count=extreme_count,
        extreme_factor=extreme_factor,
    )

    corrupted_model = pfi.declare_neuron_fi(
        function=hook,
        layer_num=[target_idx],
        batch=[0],
        dim1=dim1,
        dim2=dim2,
        dim3=dim3,
    )

    if device is not None:
        corrupted_model = corrupted_model.to(device)

    return pfi, corrupted_model


def setup_pytorchfi_bitflip_output_model(
    model: nn.Module,
    *,
    batch_size: int,
    input_shape: Sequence[int],
    target_layer_name: str,
    bit_count: int,
    bit_position: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[fault_injection, nn.Module]:
    """Setup PyTorchFI model with bit flip fault injection.

    Args:
        model: Model to inject faults into
        batch_size: Batch size for inference
        input_shape: Input tensor shape (e.g., (3, 640, 640))
        target_layer_name: Layer name to inject faults
        bit_count: Number of values to flip bits in
        bit_position: Specific bit position (0-31), or None for random
        seed: Random seed
        device: Device to run on

    Returns:
        Tuple of (fault_injection instance, corrupted model)
    """
    if seed is not None:
        seed_everything(int(seed))

    if device is not None:
        model = model.to(device)

    layer_types = [nn.Conv2d, nn.Linear]
    pfi = setup_pytorchfi(
        model,
        batch_size=batch_size,
        input_shape=input_shape,
        layer_types=layer_types,
        use_cuda=(device is not None and device.type == "cuda") if device is not None else None,
    )

    name_to_idx = build_pytorchfi_layer_name_to_index(model, layer_types)
    if target_layer_name not in name_to_idx:
        raise KeyError(
            f"target_layer_name='{target_layer_name}' not found in PyTorchFI layer map. "
            f"Example names: {list(name_to_idx.keys())[:10]}"
        )
    target_idx = int(name_to_idx[target_layer_name])

    # Provide one valid injection coordinate to satisfy bounds checks.
    layer_dim = int(pfi.get_layer_dim(target_idx))
    dim1 = [0]
    if layer_dim <= 2:
        dim2 = [None]
        dim3 = [None]
    elif layer_dim == 3:
        dim2 = [0]
        dim3 = [None]
    else:
        dim2 = [0]
        dim3 = [0]

    hook = make_bitflip_output_hook(
        pfi,
        target_layer_idx=target_idx,
        bit_count=bit_count,
        bit_position=bit_position,
    )

    corrupted_model = pfi.declare_neuron_fi(
        function=hook,
        layer_num=[target_idx],
        batch=[0],
        dim1=dim1,
        dim2=dim2,
        dim3=dim3,
    )

    if device is not None:
        corrupted_model = corrupted_model.to(device)

    return pfi, corrupted_model


class ResetEachForward(nn.Module):
    """Wrap a PyTorchFI model to reset layer cursor each forward()."""

    def __init__(self, model: nn.Module, pfi: fault_injection):
        super().__init__()
        self.model = model
        self.pfi = pfi

    def forward(self, *args, **kwargs):
        self.pfi.reset_current_layer()
        return self.model(*args, **kwargs)


