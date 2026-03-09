#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ── Config ────────────────────────────────────────────────────────────────────
NVBIT_SO=$SCRIPT_DIR/../tensor_dynamic/tensor_dynamic.so
OUTDIR=$SCRIPT_DIR/results
NUM_TRIALS=50

CORRUPT_MULT=10          # error multiplier  (scale_10)
INJECTION_MODE=2         # 2 = F32 error multiplication
TARGET_KERNEL_POS=1      # first HMMA kernel in each range
TARGET_INSTR_LIST=1      # first HMMA instruction

# Must match INJECTION_COUNTS in compare_cnn.py / compare_gemm.py
INJECTION_COUNTS=(10 100 500 1000 5000 10000)
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$OUTDIR"
cd "$SCRIPT_DIR"

# Boundary covering all kernels for these simple single-op scripts
echo "1,1000" > boundary_matrix.txt
BOUNDARY=$SCRIPT_DIR/boundary_matrix.txt

# ── Step 1: Golden (no injection, run once) ───────────────────────────────────
echo "=== Generating golden outputs ==="
python3 conv_matrix.py
mv cnn_out.pt "$OUTDIR/golden_cnn_out.pt"

python3 matrix_mult.py
mv gemm_out.pt "$OUTDIR/golden_gemm_out.pt"

echo "Golden outputs saved."

# ── Step 2: 50 trials ─────────────────────────────────────────────────────────
for TRIAL in $(seq 1 $NUM_TRIALS); do
    TRIAL_DIR="$OUTDIR/trial_$TRIAL"
    mkdir -p "$TRIAL_DIR"

    echo ""
    echo "=== Trial $TRIAL / $NUM_TRIALS ==="

    # Copy golden files into trial dir (compare scripts look for them by name)
    cp "$OUTDIR/golden_cnn_out.pt"  "$TRIAL_DIR/golden_cnn_out.pt"
    cp "$OUTDIR/golden_gemm_out.pt" "$TRIAL_DIR/golden_gemm_out.pt"

    for N in "${INJECTION_COUNTS[@]}"; do
        echo "  Injecting N=$N ..."

        # CNN
        NUM_THREADS_RECORD=$N \
        CORRUPT_MULT_LOW=$CORRUPT_MULT \
        INJECTION_MODE=$INJECTION_MODE \
        TARGET_KERNEL_POS=$TARGET_KERNEL_POS \
        TARGET_INSTR_LIST=$TARGET_INSTR_LIST \
        BOUNDARY_PATH=$BOUNDARY \
        TOOL_VERBOSE=0 \
        LD_PRELOAD=$NVBIT_SO \
        python3 conv_matrix.py
        mv cnn_out.pt "$TRIAL_DIR/${N}_cnn_out.pt"

        # GEMM
        NUM_THREADS_RECORD=$N \
        CORRUPT_MULT_LOW=$CORRUPT_MULT \
        INJECTION_MODE=$INJECTION_MODE \
        TARGET_KERNEL_POS=$TARGET_KERNEL_POS \
        TARGET_INSTR_LIST=$TARGET_INSTR_LIST \
        BOUNDARY_PATH=$BOUNDARY \
        TOOL_VERBOSE=0 \
        LD_PRELOAD=$NVBIT_SO \
        python3 matrix_mult.py
        mv gemm_out.pt "$TRIAL_DIR/${N}_gemm_out.pt"
    done

    # Run comparison scripts inside trial dir
    cp "$SCRIPT_DIR/compare_cnn.py"  "$TRIAL_DIR/"
    cp "$SCRIPT_DIR/compare_gemm.py" "$TRIAL_DIR/"
    cd "$TRIAL_DIR"
    python3 compare_cnn.py
    python3 compare_gemm.py
    cd "$SCRIPT_DIR"

    echo "  Trial $TRIAL done."
done

# ── Step 3: Aggregate over all trials and print summary table ─────────────────
echo ""
echo "=== Aggregating $NUM_TRIALS trials ==="
python3 - "$OUTDIR" "$NUM_TRIALS" <<'EOF'
import csv, sys
from pathlib import Path

outdir     = Path(sys.argv[1])
num_trials = int(sys.argv[2])

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def collect(task):
    rows = []
    for t in range(1, num_trials + 1):
        p = outdir / f"trial_{t}" / f"metrics_summary_{task}.csv"
        if not p.exists():
            print(f"  WARNING: missing {p}", file=sys.stderr)
            continue
        rows.extend(load_csv(p))
    return rows

def avg(rows, key):
    vals = [float(r[key]) for r in rows]
    return sum(vals) / len(vals)

cnn  = collect("cnn")
gemm = collect("gemm")

total    = int(cnn[0]['total_elements'])
max_n    = max(int(r['injected_events']) for r in cnn)
inj_rate = max_n / total * 100

obs_cnn  = avg(cnn,  'observability_changed_per_inject') * 100
obs_gemm = avg(gemm, 'observability_changed_per_inject') * 100
mse_cnn  = avg(cnn,  'mse_fp32')
mse_gemm = avg(gemm, 'mse_fp32')
sc_cnn   = avg(cnn,  'abs_scale_mean')
sc_gemm  = avg(gemm, 'abs_scale_mean')

# Print table to stdout
W = 35
print()
print(f"{'Metric':<{W}} {'Convolution':>14} {'GEMM':>10}")
print("-" * (W + 26))
print(f"{'Total output elements':<{W}} {total:>14,} {total:>10,}")
print(f"{'Average insertion rate (%)':<{W}} {inj_rate:>14.3f} {inj_rate:>10.3f}")
print(f"{'Observability (%)':<{W}} {obs_cnn:>14.1f} {obs_gemm:>10.1f}")
print(f"{'Mean Squared Error':<{W}} {mse_cnn:>14.3f} {mse_gemm:>10.3f}")
print(f"{'Mean Abs. Scale':<{W}} {sc_cnn:>14.3f} {sc_gemm:>10.3f}")
print()

# Save summary CSV
out_csv = outdir / "table2_summary.csv"
rows = [
    {"Metric": "Total output elements",      "Convolution": f"{total:,}",          "GEMM": f"{total:,}"},
    {"Metric": "Average insertion rate (%)", "Convolution": f"{inj_rate:.3f}",     "GEMM": f"{inj_rate:.3f}"},
    {"Metric": "Observability (%)",          "Convolution": f"{obs_cnn:.1f}",      "GEMM": f"{obs_gemm:.1f}"},
    {"Metric": "Mean Squared Error",         "Convolution": f"{mse_cnn:.3f}",      "GEMM": f"{mse_gemm:.3f}"},
    {"Metric": "Mean Abs. Scale",            "Convolution": f"{sc_cnn:.3f}",       "GEMM": f"{sc_gemm:.3f}"},
]
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["Metric", "Convolution", "GEMM"])
    w.writeheader()
    w.writerows(rows)
print(f"Summary CSV saved to {out_csv}")
EOF

echo "=== Done. Results in $OUTDIR/ ==="
