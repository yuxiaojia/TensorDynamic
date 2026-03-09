#!/bin/bash
# Figure 5 — Full pipeline: sweep + parse + plot.
# Can be run from any directory.

set -e

FIGURE5_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo "Figure 5 — Full Pipeline"
echo "========================================================================"

echo ""
echo "=== [1/2] Baseline + TensorDynamic sweep ==="
bash "$FIGURE5_DIR/sweep_baseline.sh"

echo ""
echo "=== [2/2] Parse results + generate plot ==="
bash "$FIGURE5_DIR/plot_figure5.sh"

echo ""
echo "========================================================================"
echo "Figure 5 complete."
echo "  Plots: $FIGURE5_DIR/plots/nine_point_plot.png"
echo "         $FIGURE5_DIR/plots/nine_point_plot.pdf"
echo "========================================================================"
