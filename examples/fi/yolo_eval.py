from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from fi.paths import repo_root, repo_path


def _ensure_yolov9_on_syspath(yolo_repo_path: Path | None = None) -> Path:
    """Add YOLO repository to sys.path for imports."""
    if yolo_repo_path is None:
        yroot = repo_path("yolo", "yolov9")
    else:
        yroot = yolo_repo_path

    if not yroot.exists():
        raise FileNotFoundError(f"Expected YOLO repo at: {yroot}")
    # Ensure yolov9 comes before repo root so `from models.*` resolves to yolov9/models.
    yroot_str = str(yroot)
    if sys.path[:1] != [yroot_str]:
        if yroot_str in sys.path:
            sys.path.remove(yroot_str)
        sys.path.insert(0, yroot_str)
    return yroot


@dataclass(frozen=True)
class YoloEvalConfig:
    weights: Path
    data_yaml: Path
    batch: int = 8
    imgsz: int = 640
    workers: int = 8
    device: str = "0"  # yolov9 expects device string
    max_batches: int = 0  # 0 = use all, otherwise limit to first N batches
    yolo_repo_path: Path | None = None  # Optional custom YOLO repo path


def load_detect_multibackend(cfg: YoloEvalConfig):
    _ensure_yolov9_on_syspath(cfg.yolo_repo_path)
    from models.common import DetectMultiBackend  # type: ignore
    from utils.torch_utils import select_device  # type: ignore

    dev = select_device(cfg.device, batch_size=cfg.batch)
    det = DetectMultiBackend(str(cfg.weights), device=dev, data=str(cfg.data_yaml), fp16=False)
    det.eval()
    return det


class _LimitedDataLoader:
    """Wrapper to limit dataloader to first N batches."""
    def __init__(self, dataloader, max_batches: int):
        self._dl = dataloader
        self._max = max_batches
        # Expose required attributes from underlying dataloader
        self.batch_size = getattr(dataloader, 'batch_size', None)
        self.dataset = getattr(dataloader, 'dataset', None)
    
    def __iter__(self):
        for i, batch in enumerate(self._dl):
            if i >= self._max:
                break
            yield batch
    
    def __len__(self):
        return min(len(self._dl), self._max)


def build_val_dataloader(det, cfg: YoloEvalConfig, data_dict):
    _ensure_yolov9_on_syspath(cfg.yolo_repo_path)
    from utils.dataloaders import create_dataloader  # type: ignore

    stride = int(det.stride) if hasattr(det, "stride") else 32
    dl = create_dataloader(
        data_dict["val"],
        cfg.imgsz,
        cfg.batch,
        stride,
        single_cls=False,
        pad=0.5,
        rect=True,
        workers=cfg.workers,
        min_items=0,
        prefix="val: ",
    )[0]
    
    # Limit batches if configured (helps prevent OOM on large val sets)
    if cfg.max_batches > 0:
        dl = _LimitedDataLoader(dl, cfg.max_batches)
    
    return dl


@torch.no_grad()
def evaluate_with_det_wrapper(det, *, cfg: YoloEvalConfig) -> Tuple[float, float]:
    """
    Evaluate mAP metrics using an existing DetectMultiBackend wrapper `det`.
    Returns (map50, map50_95).
    """
    _ensure_yolov9_on_syspath(cfg.yolo_repo_path)
    from utils.general import check_dataset  # type: ignore
    from val import run as yolo_val_run  # type: ignore

    data_dict = check_dataset(str(cfg.data_yaml))

    # Run validation - let val.py load the model from weights
    # Don't pass the model or dataloader, let it create them internally
    try:
        (mp, mr, map50, map5095, *_), _, _ = yolo_val_run(
            data=str(cfg.data_yaml),
            weights=str(cfg.weights),
            batch_size=cfg.batch,
            imgsz=cfg.imgsz,
            task="val",
            device=cfg.device,
            workers=cfg.workers,
            plots=False,
            save_json=False,
            half=False,
            save_txt=False,
            save_conf=False,
            save_hybrid=False,
            verbose=False,
            project=str(Path.home() / "yolo_runs" / "val"),  # Save dir
            name="exp",
            exist_ok=True,
        )
        return float(map50), float(map5095)
    finally:
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

