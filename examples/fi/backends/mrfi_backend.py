from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mrfi import MRFI, EasyConfig, add_function

from fi.utils import seed_everything

_LAST_CORRUPTED: List[Tuple[float, float]] = []


def reset_last_corrupted() -> None:
    global _LAST_CORRUPTED
    _LAST_CORRUPTED = []


def get_last_corrupted() -> List[Tuple[float, float]]:
    return _LAST_CORRUPTED


def PercentageAdditiveError(x_in, factor: float):
    """
    Custom MRFI error mode matching the legacy ErrorInjector(mode='extreme'):
      new = orig + (orig * factor * sign), sign in {-1, +1} per element
    """
    factor = float(factor)

    if torch.is_tensor(x_in):
        x = x_in.to(torch.float32).contiguous()
        device = x.device
        out_dtype = x_in.dtype
    else:
        x = torch.as_tensor(x_in, dtype=torch.float32)
        device = x.device
        out_dtype = torch.float32

    signs = (
        torch.randint(0, 2, x.shape, device=device, dtype=torch.float32) * 2 - 1
    )  # -1 or +1
    corrupted = x + (x * factor * signs)

    global _LAST_CORRUPTED
    _LAST_CORRUPTED = list(zip(x.detach().cpu().tolist(), corrupted.detach().cpu().tolist()))

    return corrupted.to(dtype=out_dtype)


def RandomBitFlip(x_in, bit_position: Optional[int] = None):
    """
    Custom MRFI error mode for random bit flips.
    Flips bits in the float32 representation.

    Args:
        x_in: Input tensor
        bit_position: Specific bit position to flip (0-31), or None for random per element
    """
    if torch.is_tensor(x_in):
        x = x_in.to(torch.float32).contiguous()
        device = x.device
        out_dtype = x_in.dtype
    else:
        x = torch.as_tensor(x_in, dtype=torch.float32)
        device = x.device
        out_dtype = torch.float32

    # Convert to int32 for bit manipulation
    x_int = x.view(torch.int32)

    # Determine bit positions
    if bit_position is not None:
        bit_pos = int(bit_position) % 32
        mask = torch.tensor(1 << bit_pos, dtype=torch.int32, device=device)
    else:
        # Random bit position per element
        bit_positions = torch.randint(0, 32, x.shape, device=device, dtype=torch.int32)
        mask = 1 << bit_positions

    # Flip bits using XOR
    flipped_int = x_int ^ mask

    # Convert back to float
    corrupted = flipped_int.view(torch.float32)

    global _LAST_CORRUPTED
    _LAST_CORRUPTED = list(zip(x.detach().cpu().tolist(), corrupted.detach().cpu().tolist()))

    return corrupted.to(dtype=out_dtype)


# Register functions with MRFI.
add_function("PercentageAdditiveError", PercentageAdditiveError)
add_function("RandomBitFlip", RandomBitFlip)


def create_mrfi_config(
    *,
    target_layer: str,
    extreme_count: int,
    extreme_factor: float,
    seed: Optional[int],
) -> EasyConfig:
    cfg = {
        "faultinject": [
            {
                # Use "activation_out" for OUTPUT-only injection (after layer computes output)
                # "activation" would inject on INPUT to the layer
                "type": "activation_out",
                "enabled": True,
                "module_name": [target_layer],
                "selector": {"method": "RandomPositionByNumber", "n": int(extreme_count)},
                "error_mode": {"method": "PercentageAdditiveError", "factor": float(extreme_factor)},
            }
        ]
    }
    if seed is not None:
        cfg["seed"] = int(seed)
    return EasyConfig(cfg)


def create_mrfi_bitflip_config(
    *,
    target_layer: str,
    bit_count: int,
    bit_position: Optional[int],
    seed: Optional[int],
) -> EasyConfig:
    """Create MRFI config for bit flip fault injection."""
    error_mode = {"method": "RandomBitFlip"}
    if bit_position is not None:
        error_mode["bit_position"] = int(bit_position)

    cfg = {
        "faultinject": [
            {
                "type": "activation_out",
                "enabled": True,
                "module_name": [target_layer],
                "selector": {"method": "RandomPositionByNumber", "n": int(bit_count)},
                "error_mode": error_mode,
            }
        ]
    }
    if seed is not None:
        cfg["seed"] = int(seed)
    return EasyConfig(cfg)


def wrap_mrfi_extreme_output_only(
    model: nn.Module,
    *,
    target_layer: str,
    extreme_count: int,
    extreme_factor: float,
    seed: Optional[int],
) -> nn.Module:
    """
    Return an MRFI-wrapped model that injects output-only corruption on `target_layer`.

    Note: some MRFI versions return an object that is *callable* but is not a `torch.nn.Module`.
    For compatibility with code that requires `nn.Module` (e.g., YOLO's DetectMultiBackend),
    we return an `nn.Module` adapter that delegates `forward()` to the MRFI object.

    Important: do NOT call `.to(...)` on the underlying MRFI object; ensure `model` is already
    on the correct device before wrapping and adapting.
    """
    if seed is not None:
        seed_everything(int(seed))
    
    # MRFI module_name uses SUBSTRING matching, but we want EXACT matching
    # to avoid accidentally matching multiple layers. Use module_fullname instead.
    config = create_mrfi_config(
        target_layer=target_layer,
        extreme_count=extreme_count,
        extreme_factor=extreme_factor,
        seed=seed,
    )
    import sys
    import logging
    # Enable MRFI's internal logging at INFO level to see if modules are found
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    
    mrfi_obj = MRFI(model, config)
    
    # Check if activation config was set on target layer
    for name, mod in model.named_modules():
        if target_layer in name and hasattr(mod, 'FI_config'):
            fi_cfg = mod.FI_config
            act = getattr(fi_cfg, 'activation', None)
            act_out = getattr(fi_cfg, 'activation_out', None)
            if act or act_out:
                print(f"[MRFI DEBUG] Layer '{name}' has FI_config: activation={len(act) if act else 0}, activation_out={len(act_out) if act_out else 0}", file=sys.stderr)
            else:
                print(f"[MRFI DEBUG] Layer '{name}' has FI_config but NO activation configs!", file=sys.stderr)
            break
    else:
        print(f"[MRFI WARNING] No layer matching '{target_layer}' has FI_config!", file=sys.stderr)
    
    return _MRFIAdapter(mrfi_obj, model)


def wrap_mrfi_bitflip_output_only(
    model: nn.Module,
    *,
    target_layer: str,
    bit_count: int,
    bit_position: Optional[int] = None,
    seed: Optional[int],
) -> nn.Module:
    """
    Return an MRFI-wrapped model that injects bit flip faults on `target_layer`.

    Args:
        model: Model to wrap (must already be on correct device)
        target_layer: Layer name to inject faults
        bit_count: Number of values to flip bits in
        bit_position: Specific bit position (0-31), or None for random
        seed: Random seed

    Returns:
        MRFI-wrapped model as nn.Module
    """
    if seed is not None:
        seed_everything(int(seed))

    config = create_mrfi_bitflip_config(
        target_layer=target_layer,
        bit_count=bit_count,
        bit_position=bit_position,
        seed=seed,
    )

    import sys
    import logging
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    mrfi_obj = MRFI(model, config)

    # Check if activation config was set on target layer
    for name, mod in model.named_modules():
        if target_layer in name and hasattr(mod, 'FI_config'):
            fi_cfg = mod.FI_config
            act = getattr(fi_cfg, 'activation', None)
            act_out = getattr(fi_cfg, 'activation_out', None)
            if act or act_out:
                print(f"[MRFI DEBUG] Layer '{name}' has FI_config: activation={len(act) if act else 0}, activation_out={len(act_out) if act_out else 0}", file=sys.stderr)
            else:
                print(f"[MRFI DEBUG] Layer '{name}' has FI_config but NO activation configs!", file=sys.stderr)
            break
    else:
        print(f"[MRFI WARNING] No layer matching '{target_layer}' has FI_config!", file=sys.stderr)

    return _MRFIAdapter(mrfi_obj, model)


class _MRFIAdapter(nn.Module):
    """
    Minimal adapter to present an MRFI callable as a torch.nn.Module.
    """

    def __init__(self, mrfi_obj, wrapped_model: nn.Module):
        super().__init__()
        # Expose the wrapped model under `.module` to match the common DataParallel/DDP
        # convention used inside YOLOv9 (it accesses `model.module.names` in val.py).
        # This also ensures `.parameters()` is non-empty.
        self.module = wrapped_model
        self._mrfi = mrfi_obj

    def forward(self, *args, **kwargs):
        out = self._mrfi(*args, **kwargs)
        # Some MRFI versions return additional metadata / activation traces alongside
        # the model output. We must return the *model output tensor* expected by downstream
        # evaluation loops (classifier logits: [B, C], YOLO preds: [B, N, >=6]).
        if isinstance(out, (tuple, list)):
            tensors = []

            def _collect(x):
                if torch.is_tensor(x):
                    tensors.append(x)
                elif isinstance(x, (list, tuple)):
                    for y in x:
                        _collect(y)

            _collect(out)
            if tensors:
                # Heuristic: YOLO predictions are typically a 3D tensor (bs, n, no>=6).
                for t in tensors:
                    try:
                        if t.ndim == 3 and t.shape[-1] >= 6:
                            return t
                    except Exception:
                        pass
                # Heuristic: classifier logits are typically a 2D tensor (bs, classes).
                for t in reversed(tensors):
                    try:
                        if t.ndim == 2:
                            return t
                    except Exception:
                        pass
                # Fallback: return the last tensor seen (often the final output).
                return tensors[-1]

        return out


