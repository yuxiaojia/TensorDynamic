#!/usr/bin/env python3
"""
Paper-ready 4-panel figure:
Sparsity of Tensor Core destination registers after computation.

Reads these CSVs (same folder as this script):
  - resnet_reg_dist_distribution.csv
  - mobilenet_reg_dist_distribution.csv
  - shufflenet_reg_dist_distribution.csv
  - yolo_reg_dist_distribution.csv

X-axis:
  Fraction of zero values in Tensor Core destination registers

Y-axis:
  Percentage of Tensor Core kernels

Output:
  reg_dist_plots/combined_sparsity_dest_zero_4panel.png  (300 dpi)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 19
plt.rcParams['ytick.labelsize'] = 19

INPUTS = [
    ("resnet_reg_dist_distribution.csv", "ResNet20"),
    ("mobilenet_reg_dist_distribution.csv", "MobileNetv2"),
    ("shufflenet_reg_dist_distribution.csv", "ShuffleNetv2"),
    ("yolo_reg_dist_distribution.csv", "YOLOv9"),
]

OUTDIR = "reg_dist_plots"
OUTFILE = "combined_sparsity_dest_zero_4panel.png"
DPI = 300


def pct_to_float(x) -> float:
    if pd.isna(x):
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    return float(str(x).strip().replace("%", ""))


def main() -> None:
    series = []

    for csv_name, display_name in INPUTS:
        p = Path(csv_name)
        if not p.exists():
            raise FileNotFoundError(f"Missing input CSV: {p.resolve()}")

        df = pd.read_csv(p)
        val_col = "Count" if "Count" in df.columns else "Percentage"
        if "Bin_Range" not in df.columns or val_col not in df.columns:
            raise ValueError(
                f"{csv_name}: expected columns 'Bin_Range' and 'Count' (or 'Percentage'). "
                f"Found: {list(df.columns)}"
            )

        # extract the average row written by parse_reg_dist.py
        avg_row = df[df["Bin_Range"] == "Average"]
        avg_zeros = float(avg_row[val_col].values[0]) if not avg_row.empty else 0.0

        # bin rows only
        df_bins = df[df["Bin_Range"] != "Average"]
        x = df_bins["Bin_Range"].astype(str).tolist()
        y = df_bins[val_col].apply(pct_to_float).astype(float).tolist()
        series.append((display_name, x, y, avg_zeros))

    fig, axes = plt.subplots(2, 2, figsize=(16, 13), sharey=True)
    axes = axes.flatten()

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    line_color = "#4E79A7"

    for ax, label, (name, x, y, avg_zeros) in zip(axes, panel_labels, series):
        ax.plot(range(len(x)), y, color=line_color, marker='o', linewidth=2.5,
                markersize=7, alpha=0.9)
        ax.fill_between(range(len(x)), y, alpha=0.15, color=line_color)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x)

        # Draw average zero% as a vertical dashed line
        # avg_zeros is a percentage (0–100); bins are 0–9 on the x-axis (each covers 10%)
        avg_x = avg_zeros / 10.0
        ax.axvline(avg_x, color="crimson", linestyle="--", linewidth=2.0,
                   label=f"Avg: {avg_zeros:.1f}%")

        ax.set_title(f"{label} {name}", fontsize=31, fontweight="bold", pad=15)
        ax.tick_params(axis="x", rotation=45, labelsize=20)
        ax.tick_params(axis="y", labelsize=20)

        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")
        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")

        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        ax.set_axisbelow(True)

        legend = ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=25)
        for text in legend.get_texts():
            text.set_fontweight('bold')

    fig.supxlabel(
        "Fraction of Zero Values in HMMA Destination Registers",
        fontsize=35, fontweight="bold", y=0.065, x=0.5
    )
    fig.supylabel(
        "Number of Kernel Invocations",
        fontsize=35, fontweight="bold", x=0.055
    )

    fig.tight_layout(rect=[0.04, 0.04, 0.99, 1])

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    out_png = outdir / OUTFILE
    out_pdf = outdir / OUTFILE.replace('.png', '.pdf')
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Wrote: {out_png.resolve()}")
    print(f"Wrote: {out_pdf.resolve()}")


if __name__ == "__main__":
    main()
