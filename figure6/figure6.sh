#!/bin/bash
# Collect HMMA register zero/non-zero distribution for all models (Figure 6).
# Can be run from any directory.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"  # TensorDynamic/

TOOL_DIR="$ROOT_DIR/hmma_reg_distribution"
TOOL_SO="$TOOL_DIR/hmma_reg_distribution.so"
EXAMPLES_DIR="$ROOT_DIR/examples"
YOLOV9_DIR="$EXAMPLES_DIR/yolo/yolov9"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR"

# Build the tool if not already built
if [ ! -f "$TOOL_SO" ]; then
    echo "Building hmma_reg_distribution..."
    (cd "$TOOL_DIR" && make clean && make)
fi

echo "Running resnet20..."
LD_PRELOAD="$TOOL_SO" python3 "$EXAMPLES_DIR/eval_resnet20.py" \
    > "$RESULTS_DIR/resnet20.txt" 2>&1
echo "  -> $RESULTS_DIR/resnet20.txt"

echo "Running mobilenet..."
LD_PRELOAD="$TOOL_SO" python3 "$EXAMPLES_DIR/eval_mobilenet.py" \
    > "$RESULTS_DIR/mobilenet.txt" 2>&1
echo "  -> $RESULTS_DIR/mobilenet.txt"

echo "Running shufflenet..."
LD_PRELOAD="$TOOL_SO" python3 "$EXAMPLES_DIR/eval_shufflenet.py" \
    > "$RESULTS_DIR/shufflenet.txt" 2>&1
echo "  -> $RESULTS_DIR/shufflenet.txt"

echo "Running yolo..."
(cd "$YOLOV9_DIR" && LD_PRELOAD="$TOOL_SO" python3 eval_yolo.py) \
    > "$RESULTS_DIR/yolo.txt" 2>&1
echo "  -> $RESULTS_DIR/yolo.txt"

# Parse each output file into a binned CSV
echo "Parsing results..."
python3 "$SCRIPT_DIR/parse_reg_dist.py" \
    "$RESULTS_DIR/resnet20.txt"   "$SCRIPT_DIR/resnet_reg_dist_distribution.csv"
python3 "$SCRIPT_DIR/parse_reg_dist.py" \
    "$RESULTS_DIR/mobilenet.txt"  "$SCRIPT_DIR/mobilenet_reg_dist_distribution.csv"
python3 "$SCRIPT_DIR/parse_reg_dist.py" \
    "$RESULTS_DIR/shufflenet.txt" "$SCRIPT_DIR/shufflenet_reg_dist_distribution.csv"
python3 "$SCRIPT_DIR/parse_reg_dist.py" \
    "$RESULTS_DIR/yolo.txt"       "$SCRIPT_DIR/yolo_reg_dist_distribution.csv"

# Generate the 4-panel figure
echo "Plotting..."
(cd "$SCRIPT_DIR" && python3 plot_reg_dist.py)

echo ""
echo "Done. Results in $RESULTS_DIR/"
echo "Figure in $SCRIPT_DIR/reg_dist_plots/"
