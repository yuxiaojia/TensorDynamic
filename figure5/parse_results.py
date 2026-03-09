#!/usr/bin/env python3
"""
Figure 5 — parse raw sweep CSVs into pivot tables for plotting.

Reads from results_baseline/:
  {model}_pytorchfi_mult.csv    -> baseline (average across layers at count=500, factor=1000)
  {model}_tensordynamic_mult.csv -> 9-point sweep averaged across kernels (kernel1/3/5)

Outputs to csv/:
  accuracy_pivot_resnet.csv
  accuracy_pivot_mobile.csv
  accuracy_pivot_shuff.csv
  mAP50_pivot_yolo.csv
  baselines.csv
"""

import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results_baseline")
OUT_DIR = Path("csv")
OUT_DIR.mkdir(exist_ok=True)

# (model key, short name for output file, metric label)
MODELS = [
    ("resnet20",   "resnet", "accuracy"),
    ("mobilenet",  "mobile", "accuracy"),
    ("shufflenet", "shuff",  "accuracy"),
    ("yolo",       "yolo",   "mAP50"),
]

baselines = {}

for model_key, short, metric in MODELS:
    # ── PyTorchFI baseline: average across the 3 layers ──────────────────────
    pfi_path = RESULTS_DIR / f"{model_key}_pytorchfi_mult.csv"
    if not pfi_path.exists():
        print(f"WARNING: {pfi_path} not found — skipping baseline for {model_key}")
        baselines[model_key] = float("nan")
    else:
        pfi = pd.read_csv(pfi_path)
        baseline_val = pfi["accuracy"].mean()
        baselines[model_key] = baseline_val
        print(f"{model_key} PyTorchFI baseline (avg across layers): {baseline_val:.4f}")

    # ── TensorDynamic sweep: average across kernels (kernel1/3/5) ─────────────
    td_path = RESULTS_DIR / f"{model_key}_tensordynamic_mult.csv"
    if not td_path.exists():
        print(f"WARNING: {td_path} not found — skipping pivot for {model_key}")
        continue

    # Try reading from individual txt files first (more reliable than CSV)
    txt_dir = RESULTS_DIR / f"{model_key}_tensordynamic_mult"
    rows = []
    if txt_dir.exists():
        import re
        for txt_file in txt_dir.glob("nt*_cl*_k*_i*.txt"):
            m = re.search(r"nt(\d+)_cl(\d+)_k(\d+)_i(\d+)\.txt", txt_file.name)
            if not m:
                continue
            nt, cl, k = int(m.group(1)), int(m.group(2)), int(m.group(3))
            text = txt_file.read_text()
            acc_m = re.search(r"Top-1 Accuracy:\s*([\d.]+)", text)
            if not acc_m:
                acc_m = re.search(r"^mAP50:\s*([\d.]+)", text, re.MULTILINE)
            if acc_m:
                rows.append({"count": nt, "factor": cl, "kernel": f"kernel{k}",
                             "accuracy": float(acc_m.group(1))})
            else:
                print(f"  WARNING: no accuracy found in {txt_file.name}")
        if rows:
            td = pd.DataFrame(rows)
            print(f"  Read {len(rows)} runs from {txt_dir.name}/")
        else:
            td = pd.read_csv(td_path, on_bad_lines="skip")
            td["accuracy"] = pd.to_numeric(td["accuracy"], errors="coerce")
            td = td.dropna(subset=["accuracy"])
    else:
        td = pd.read_csv(td_path, on_bad_lines="skip")
        td["accuracy"] = pd.to_numeric(td["accuracy"], errors="coerce")
        td = td.dropna(subset=["accuracy"])

    # Average accuracy across kernel1, kernel3, kernel5 for each (count, factor)
    agg = (td.groupby(["count", "factor"])["accuracy"]
             .mean()
             .reset_index())

    # Pivot: rows = count, columns = factor
    pivot = agg.pivot(index="count", columns="factor", values="accuracy")
    pivot.columns = [f"mult={int(c)}" for c in pivot.columns]
    pivot = pivot.reset_index()

    out_name = f"mAP50_pivot_{short}.csv" if metric == "mAP50" else f"accuracy_pivot_{short}.csv"
    out_path = OUT_DIR / out_name
    pivot.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

# ── Baselines CSV ─────────────────────────────────────────────────────────────
baseline_df = pd.DataFrame([
    {"model": k, "baseline": v} for k, v in baselines.items()
])
baseline_df.to_csv(OUT_DIR / "baselines.csv", index=False)
print(f"Saved {OUT_DIR}/baselines.csv")
print("\nDone.")
