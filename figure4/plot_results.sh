#!/bin/bash
# Parse sweep results and generate Figure 4 (8-panel plot).
# Run from any directory.

set -e

FIGURE4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSORDYNAMIC_DIR="$(dirname "$FIGURE4_DIR")"
EXAMPLES_DIR="$TENSORDYNAMIC_DIR/examples"
YOLOV9_DIR="$EXAMPLES_DIR/yolo/yolov9"

PFI_DIR="$FIGURE4_DIR/results_pytorchfi_mrfi"
TD_DIR="$FIGURE4_DIR/results_tensordynamic"

echo "========================================================================"
echo "Figure 4 — Baselines + Parse + Plot"
echo "PyTorchFI/MRFI results: $PFI_DIR"
echo "TensorDynamic results:  $TD_DIR"
echo "Output:                 $FIGURE4_DIR"
echo "========================================================================"

# Helper: run an eval script and extract accuracy/mAP50
run_baseline() {
    local script="$1"
    local workdir="$2"
    local OUT
    OUT=$(cd "$workdir" && python3 "$script" 2>&1)
    local ACC
    ACC=$(echo "$OUT" | grep "Top-1 Accuracy:" | awk '{print $3}' | tr -d '%')
    [ -z "$ACC" ] && ACC=$(echo "$OUT" | grep "^mAP50:" | awk '{print $2}')
    echo "$ACC"
}

echo ""
echo "--- Step 0: Collecting baselines (no fault injection) ---"
mkdir -p "$FIGURE4_DIR/csv"
BASELINES_CSV="$FIGURE4_DIR/csv/baselines.csv"
echo "model,baseline" > "$BASELINES_CSV"

echo "  Running eval_resnet20.py ..."
RESNET_BASE=$(run_baseline "$EXAMPLES_DIR/eval_resnet20.py" "$EXAMPLES_DIR")
echo "resnet,$RESNET_BASE" >> "$BASELINES_CSV"
echo "    resnet -> $RESNET_BASE"

echo "  Running eval_mobilenet.py ..."
MOBILE_BASE=$(run_baseline "$EXAMPLES_DIR/eval_mobilenet.py" "$EXAMPLES_DIR")
echo "mobilenet,$MOBILE_BASE" >> "$BASELINES_CSV"
echo "    mobilenet -> $MOBILE_BASE"

echo "  Running eval_shufflenet.py ..."
SHUFFLE_BASE=$(run_baseline "$EXAMPLES_DIR/eval_shufflenet.py" "$EXAMPLES_DIR")
echo "shufflenet,$SHUFFLE_BASE" >> "$BASELINES_CSV"
echo "    shufflenet -> $SHUFFLE_BASE"

echo "  Running eval_yolo.py ..."
YOLO_BASE=$(run_baseline "$YOLOV9_DIR/eval_yolo.py" "$YOLOV9_DIR")
echo "yolo,$YOLO_BASE" >> "$BASELINES_CSV"
echo "    yolo -> $YOLO_BASE"

echo "  Baselines written to $BASELINES_CSV"

echo ""
echo "--- Step 1: Parsing results ---"
python3 "$FIGURE4_DIR/parse_results.py" \
    --pytorchfi_mrfi_dir "$PFI_DIR" \
    --tensordynamic_dir  "$TD_DIR" \
    --outdir             "$FIGURE4_DIR" \
    --exclude_factors    10000

echo ""
echo "--- Step 2: Plotting 8 panels ---"
cd "$FIGURE4_DIR"
python3 plot_8_panels.py

echo ""
echo "========================================================================"
echo "Done."
echo "  CSVs:  $FIGURE4_DIR/csv/"
echo "  Plots: $FIGURE4_DIR/plots/eight_plots.png / eight_plots.pdf"
echo "========================================================================"
