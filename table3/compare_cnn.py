# sweep_compare_mul_fp32_to_csv.py
import torch
import csv
from pathlib import Path

# ===================== FIXED SETTINGS =====================
TASK_NAME = "cnn"  # used in filenames + output csv name

GOLDEN_FILE = f"golden_{TASK_NAME}_out.pt"
INJECTION_COUNTS = [10, 100, 500, 1000, 5000, 10000]
TEST_FILE_PATTERN = "{N}_" + TASK_NAME + "_out.pt"

OUT_CSV = f"metrics_summary_{TASK_NAME}.csv"

# scaling bucket config (optional, but useful)
TARGETS = [0.5, 1, 2, 10, 100]
TOL = 0.05  # ±5%
# ==========================================================


def safe_load_pt(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p.resolve()}")
    return torch.load(p)


def main():
    golden = safe_load_pt(GOLDEN_FILE).float()
    g = golden.flatten()
    num_total = g.numel()

    rows = []

    for N in INJECTION_COUNTS:
        test_file = TEST_FILE_PATTERN.format(N=N)
        test = safe_load_pt(test_file).float()
        t = test.flatten()

        assert t.shape == g.shape, f"Shape mismatch for {test_file}: {t.shape} vs {g.shape}"

        # diff + changed_count
        diff = t - g
        abs_diff = diff.abs()
        changed_mask = (abs_diff != 0)
        changed_count = int(changed_mask.sum().item())

        # observability
        observability = changed_count / float(N)

        # MSE over all elements (FP32)
        mse = float((diff * diff).mean().item())

        # Scaling (FP32): scale = test/golden, skip golden==0
        mask_valid = (g != 0)
        skipped_golden_zero = int((~mask_valid).sum().item())
        valid_count = int(mask_valid.sum().item())

        scale = t[mask_valid] / g[mask_valid]
        is_inf = torch.isinf(scale)
        is_nan = torch.isnan(scale)
        is_finite = ~(is_inf | is_nan)

        inf_scale_count = int(is_inf.sum().item())
        nan_scale_count = int(is_nan.sum().item())
        finite_scale_count = int(is_finite.sum().item())

        finite_scale = scale[is_finite]
        abs_scale = finite_scale.abs()

        # How many finite scale entries are exactly 1?
        # (for many cases this will equal "unchanged among valid", but not always)
        scale_eq_1 = int((finite_scale == 1.0).sum().item())
        scale_ne_1 = int(finite_scale.numel() - scale_eq_1)

        # Scale stats (guard empty)
        if finite_scale.numel() > 0:
            scale_min = float(abs_scale.min().item())
            scale_median = float(abs_scale.median().item())
            scale_mean = float(abs_scale.mean().item())
            scale_max = float(abs_scale.max().item())
            scale_std = float(abs_scale.std().item())
        else:
            scale_min = scale_median = scale_mean = scale_max = scale_std = float("nan")

        # Bucket counts near common factors (finite only)
        bucket_counts = {}
        if finite_scale.numel() > 0:
            for s in TARGETS:
                cnt = int(((abs_scale - s).abs() <= (TOL * s)).sum().item())
                bucket_counts[f"count_near_{s}x"] = cnt
        else:
            for s in TARGETS:
                bucket_counts[f"count_near_{s}x"] = 0

        row = {
            "task": TASK_NAME,
            "injected_events": N,
            "total_elements": num_total,

            "changed_count": changed_count,
            "observability_changed_per_inject": observability,

            "mse_fp32": mse,

            "skipped_golden_zero": skipped_golden_zero,
            "valid_nonzero_golden": valid_count,

            "inf_scale_count": inf_scale_count,
            "nan_scale_count": nan_scale_count,
            "finite_scale_count": finite_scale_count,

            "scale_eq_1_exact": scale_eq_1,
            "scale_ne_1_exact": scale_ne_1,

            "abs_scale_min": scale_min,
            "abs_scale_median": scale_median,
            "abs_scale_mean": scale_mean,
            "abs_scale_max": scale_max,
            "abs_scale_std": scale_std,
        }

        row.update(bucket_counts)
        rows.append(row)

        print(f"[{TASK_NAME} N={N}] changed={changed_count}, obs={observability:.4f}, mse={mse:.6g}, "
              f"scale_max={scale_max:.6g}, near100x={row.get('count_near_100x', 0)}")

    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote CSV: {Path(OUT_CSV).resolve()}")


if __name__ == "__main__":
    main()
