from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from fi.layers import list_leaf_scopes
from fi.utils import amp_autocast, seed_everything


@dataclass(frozen=True)
class ExtremeSweepGrid:
    counts: Sequence[int]
    factors: Sequence[float]


@torch.no_grad()
def sweep_classifier_to_csv(
    *,
    model_name: str,
    model_factory: Callable[[torch.device], nn.Module],
    loader_factory: Callable[[], Iterable],
    device: torch.device,
    backend_name: str,
    wrap_for_layer: Callable[[nn.Module, str, int, float, int], nn.Module],
    grid: ExtremeSweepGrid,
    seed: int,
    csv_path: Path,
    layers: Optional[List[str]] = None,
    print_details: bool = False,
) -> None:
    """
    Write one CSV with header: ["model","injection",<layer1>,<layer2>,...]
    Each cell is top-1 accuracy (%) under FI for that (count,factor) and layer.
    """
    loader = loader_factory()
    # Build once for layer discovery if not provided.
    if layers is None:
        layers = list_leaf_scopes(model_factory(device))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["model", "injection"] + layers

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        f.flush()

        for count in grid.counts:
            for factor in grid.factors:
                label = f"extreme,count={count},factor={factor}"
                row = [model_name, label]

                for layer in layers:
                    try:
                        seed_everything(int(seed))
                        model = model_factory(device)
                        fi_model = wrap_for_layer(model, layer, int(count), float(factor), int(seed))
                        total = 0
                        correct = 0
                        amp_ctx = amp_autocast(device)
                        for x, y in loader:
                            x = x.to(device, non_blocking=True)
                            y = y.to(device, non_blocking=True)
                            with amp_ctx:
                                logits = fi_model(x)
                            pred = logits.argmax(1)
                            total += y.size(0)
                            correct += (pred == y).sum().item()

                        acc = 100.0 * correct / max(total, 1)
                        row.append(round(acc, 2))
                        if print_details:
                            print(
                                f"[{backend_name}] model={model_name} layer={layer} count={count} "
                                f"factor={factor} seed={seed} acc={acc:.2f}%"
                            )
                    except Exception as e:
                        row.append(float("nan"))
                        print(
                            f"[{backend_name} FAIL] model={model_name} layer={layer} "
                            f"c={count} f={factor} seed={seed}: {e}",
                            file=sys.stderr,
                        )
                w.writerow(row)
                f.flush()


def sweep_yolo_to_csv(
    *,
    model_label: str,
    backend_name: str,
    grid: ExtremeSweepGrid,
    seed: int,
    csv_path: Path,
    layer_names: List[str],
    eval_one: Callable[[str, int, float, int], Tuple[float, float]],
) -> None:
    """
    Write YOLO CSV with header: ["model","layer","injection","map50","map50_95"].
    One row per (seed,count,factor,layer).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["model", "layer", "injection", "map50", "map50_95"]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        f.flush()

        printed_traceback = False
        for count in grid.counts:
            for factor in grid.factors:
                label = f"extreme,count={count},factor={factor}"
                for layer in layer_names:
                    try:
                        map50, map5095 = eval_one(layer, int(count), float(factor), int(seed))
                        w.writerow([model_label, layer, label, round(map50, 4), round(map5095, 4)])
                    except Exception as e:
                        if not printed_traceback:
                            import traceback

                            traceback.print_exc()
                            printed_traceback = True
                        print(
                            f"[{backend_name} YOLO FAIL] layer={layer} c={count} f={factor} seed={seed}: "
                            f"{type(e).__name__}: {e!r}",
                            file=sys.stderr,
                        )
                        w.writerow([model_label, layer, label, float("nan"), float("nan")])
                    f.flush()


