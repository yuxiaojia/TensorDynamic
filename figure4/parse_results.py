#!/usr/bin/env python3
"""
Parse raw sweep results into 8 summary CSVs for plot_8_panels.py.

Usage:
    # Both backends run separately into their own folders:
    python parse_results.py --pytorchfi_mrfi_dir figure4/results_pytorchfi_mrfi \
                            --tensordynamic_dir  figure4/results_tensordynamic \
                            --outdir             figure4/

    # Both backends in the same folder:
    python parse_results.py --pytorchfi_mrfi_dir results_combined/ \
                            --tensordynamic_dir  results_combined/

results_pytorchfi_mrfi/ contains:
    resnet20_pytorchfi_mult.csv, resnet20_mrfi_mult.csv  (timestamp,layer,count,factor,accuracy)
    resnet20_pytorchfi_bit.csv,  resnet20_mrfi_bit.csv   (timestamp,layer,count,accuracy)
    mobilenet_*, shufflenet_*, yolo_*  (same pattern)

results_tensordynamic/ contains:
    tensordynamic_resnet20_mult/   (mult_{NT}_{CL}_{K}_{I}.txt)
    tensordynamic_resnet20_bit/    (bit_{NT}_{K}_{I}.txt)
    tensordynamic_mobilenet_*/
    tensordynamic_shufflenet_*/
    tensordynamic_yolo_*/

Output (8 CSVs, read by plot_8_panels.py):
    resnet_mult.csv, resnet_bitflip.csv
    mobilenet_mult.csv, mobilenet_bitflip.csv
    shufflenet_mult.csv, shufflenet_bitflip.csv
    yolo_mult.csv, yolo_bitflip.csv

Each output CSV: method,count,accuracy
  - accuracy is averaged across all factors and layers for each count
  - For TensorDynamic mult: count=NUM_THREADS_RECORD, averaged over CORRUPT_MULT_LOW/K/I
  - For TensorDynamic bit:  count=NUM_THREADS_RECORD, averaged over K/I
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


ACC_PAT = re.compile(r"Top-1 Accuracy:\s+([\d.]+)%")
MAP_PAT  = re.compile(r"^mAP50:\s+([\d.]+)", re.MULTILINE)


def parse_txt(path: Path) -> float | None:
    text = path.read_text(errors="replace")
    m = ACC_PAT.search(text) or MAP_PAT.search(text)
    return float(m.group(1)) if m else None


def load_pytorchfi_mrfi_mult(csv_path: Path, method: str, exclude_factors: set) -> pd.DataFrame:
    """Read a mult CSV and return rows averaged across factor and layer per count."""
    df = pd.read_csv(csv_path)
    df = df[df["accuracy"] != -1]
    if exclude_factors:
        df = df[~df["factor"].isin(exclude_factors)]
    agg = df.groupby("count")["accuracy"].mean().reset_index()
    agg["method"] = method
    return agg[["method", "count", "accuracy"]]


def load_pytorchfi_mrfi_bit(csv_path: Path, method: str) -> pd.DataFrame:
    """Read a bit CSV and return rows averaged across layer per count."""
    df = pd.read_csv(csv_path)
    df = df[df["accuracy"] != -1]
    agg = df.groupby("count")["accuracy"].mean().reset_index()
    agg["method"] = method
    return agg[["method", "count", "accuracy"]]


def load_tensordynamic_mult(results_dir: Path, label: str, exclude_factors: set) -> pd.DataFrame:
    """Parse tensordynamic_<label>_mult/ txt files.
    Filename: mult_{NT}_{CL}_{K}_{I}.txt  → count=NT, averaged over CL/K/I.
    """
    subdir = results_dir / f"tensordynamic_{label}_mult"
    rows = []
    for f in subdir.glob("mult_*.txt"):
        parts = f.stem.split("_")  # ['mult', NT, CL, K, I]
        if len(parts) != 5:
            continue
        nt = int(parts[1])
        cl = int(parts[2])
        if cl in exclude_factors:
            continue
        acc = parse_txt(f)
        if acc is not None:
            rows.append({"count": nt, "accuracy": acc})
    if not rows:
        return pd.DataFrame(columns=["method", "count", "accuracy"])
    df = pd.DataFrame(rows)
    agg = df.groupby("count")["accuracy"].mean().reset_index()
    agg["method"] = "nvbit"
    return agg[["method", "count", "accuracy"]]


def load_tensordynamic_bit(results_dir: Path, label: str) -> pd.DataFrame:
    """Parse tensordynamic_<label>_bit/ txt files.
    Filename: bit_{NT}_{K}_{I}.txt  → count=NT, averaged over K/I.
    """
    subdir = results_dir / f"tensordynamic_{label}_bit"
    rows = []
    for f in subdir.glob("bit_*.txt"):
        parts = f.stem.split("_")  # ['bit', NT, K, I]
        if len(parts) != 4:
            continue
        nt = int(parts[1])
        acc = parse_txt(f)
        if acc is not None:
            rows.append({"count": nt, "accuracy": acc})
    if not rows:
        return pd.DataFrame(columns=["method", "count", "accuracy"])
    df = pd.DataFrame(rows)
    agg = df.groupby("count")["accuracy"].mean().reset_index()
    agg["method"] = "nvbit"
    return agg[["method", "count", "accuracy"]]


def build_summary(pfi_dir: Path, td_dir: Path, model_key: str, sweep_prefix: str, typ: str, exclude_factors: set) -> pd.DataFrame:
    """
    pfi_dir:      directory containing PyTorchFI/MRFI CSVs
    td_dir:       directory containing tensordynamic_*/ subdirs
    model_key:    'resnet20', 'mobilenet', 'shufflenet', 'yolo'
    sweep_prefix: used in CSV filenames (same as model_key)
    typ:          'mult' or 'bit'
    """
    frames = []

    if typ == "mult":
        for method in ["pytorchfi", "mrfi"]:
            csv = pfi_dir / f"{sweep_prefix}_{method}_mult.csv"
            if csv.exists():
                frames.append(load_pytorchfi_mrfi_mult(csv, method, exclude_factors))
            else:
                print(f"  [warn] missing {csv}", file=sys.stderr)

        td_subdir = td_dir / f"tensordynamic_{model_key}_mult"
        if td_subdir.exists():
            frames.append(load_tensordynamic_mult(td_dir, model_key, exclude_factors))
        else:
            print(f"  [warn] missing {td_subdir}/", file=sys.stderr)

    else:  # bit
        for method in ["pytorchfi", "mrfi"]:
            csv = pfi_dir / f"{sweep_prefix}_{method}_bit.csv"
            if csv.exists():
                frames.append(load_pytorchfi_mrfi_bit(csv, method))
            else:
                print(f"  [warn] missing {csv}", file=sys.stderr)

        td_subdir = td_dir / f"tensordynamic_{model_key}_bit"
        if td_subdir.exists():
            frames.append(load_tensordynamic_bit(td_dir, model_key))
        else:
            print(f"  [warn] missing {td_subdir}/", file=sys.stderr)

    if not frames:
        return pd.DataFrame(columns=["method", "count", "accuracy"])
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorchfi_mrfi_dir", required=True,
                        help="Directory containing PyTorchFI/MRFI result CSVs (results_pytorchfi_mrfi/)")
    parser.add_argument("--tensordynamic_dir", required=True,
                        help="Directory containing TensorDynamic result subdirs (results_tensordynamic/)")
    parser.add_argument("--outdir", default=".",
                        help="Output directory for summary CSVs (default: current directory)")
    parser.add_argument("--exclude_factors", nargs="*", type=int, default=[],
                        help="Corruption factor values to exclude (e.g. --exclude_factors 10000)")
    args = parser.parse_args()

    pfi_dir = Path(args.pytorchfi_mrfi_dir)
    td_dir  = Path(args.tensordynamic_dir)
    outdir  = Path(args.outdir) / "csv"
    outdir.mkdir(parents=True, exist_ok=True)
    exclude_factors = set(args.exclude_factors)
    if exclude_factors:
        print(f"Excluding factors: {sorted(exclude_factors)}")

    # (output_prefix, model_key, csv_sweep_prefix)
    models = [
        ("resnet",     "resnet20",   "resnet20"),
        ("mobilenet",  "mobilenet",  "mobilenet"),
        ("shufflenet", "shufflenet", "shufflenet"),
        ("yolo",       "yolo",       "yolo"),
    ]

    for out_prefix, model_key, sweep_prefix in models:
        for typ, out_suffix in [("mult", "mult"), ("bit", "bitflip")]:
            print(f"Processing {model_key} {typ}...")
            df = build_summary(pfi_dir, td_dir, model_key, sweep_prefix, typ, exclude_factors)
            out_csv = outdir / f"{out_prefix}_{out_suffix}.csv"
            df.to_csv(out_csv, index=False)
            print(f"  -> {out_csv}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
