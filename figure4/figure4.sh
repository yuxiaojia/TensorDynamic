#!/bin/bash
# Figure 4 — Full pipeline: FI sweeps + baseline collection + parse + plot.
# Can be run from any directory.

set -e

FIGURE4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo "Figure 4 — Full Pipeline"
echo "========================================================================"

echo ""
echo "=== [1/3] PyTorchFI & MRFI sweep ==="
bash "$FIGURE4_DIR/sweep_pytorchfi_mrfi.sh"

echo ""
echo "=== [2/3] TensorDynamic (NVBit) sweep ==="
bash "$FIGURE4_DIR/sweep_tensordynamic.sh"

echo ""
echo "=== [3/3] Collect baselines + parse + plot ==="
bash "$FIGURE4_DIR/plot_results.sh"

echo ""
echo "========================================================================"
echo "Figure 4 complete."
echo "  Plots: $FIGURE4_DIR/plots/eight_plots.png / eight_plots.pdf"
echo "========================================================================"
