#!/bin/bash
# Figure 5 — parse results and generate plot.
# Run from the TensorDynamic/ directory or any directory.

set -e

FIGURE5_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$FIGURE5_DIR"

echo "=== [1/2] Parsing results ==="
python parse_results.py

echo ""
echo "=== [2/2] Generating plot ==="
python plot.py

echo ""
echo "Done. Outputs:"
echo "  CSVs:  $FIGURE5_DIR/csv/"
echo "  Plots: $FIGURE5_DIR/plots/nine_point_plot.png"
echo "         $FIGURE5_DIR/plots/nine_point_plot.pdf"
