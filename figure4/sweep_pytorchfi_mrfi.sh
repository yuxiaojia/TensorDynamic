#!/bin/bash
# Figure 4 — PyTorchFI & MRFI fault injection sweep for all four models.
# Can be run from any directory.

set -o pipefail

FIGURE4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSORDYNAMIC_DIR="$(dirname "$FIGURE4_DIR")"
EXAMPLES_DIR="$TENSORDYNAMIC_DIR/examples"
YOLOV9_DIR="$EXAMPLES_DIR/yolo/yolov9"

# Allow OUTPUT_DIR to be passed in (used when called from sweep_all.sh)
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$FIGURE4_DIR/results_pytorchfi_mrfi"
    mkdir -p "$OUTPUT_DIR"
fi
GLOBAL_LOG="$OUTPUT_DIR/sweep.log"

echo "========================================================================"
echo "PyTorchFI & MRFI Sweep — all models"
echo "Examples: $EXAMPLES_DIR"
echo "Output:   $OUTPUT_DIR"
echo "========================================================================"

# ────────────────────────────────────────────────────────────────────────────
# Parameter sets
# ────────────────────────────────────────────────────────────────────────────
COUNTS=(10 50 100 500 1000)
FACTORS=(10 50 100 1000)
BIT_COUNTS=(10 50 100 500 1000)

# 1st, 3rd, and 5th conv layers (index 0, 2, and 4 from each model's layer list)
LAYERS_RESNET=("conv1" "layer1.0.conv2" "layer1.1.conv2")
LAYERS_MOBILENET=("features.0.0" "features.1.conv.1" "features.2.conv.1.0")
LAYERS_SHUFFLE=("conv1.0" "stage2.0.branch1.2" "stage2.0.branch2.3")
LAYERS_YOLO=("model.0.conv" "model.2.cv1.conv" "model.2.cv2.0.m.0.cv1.conv")

# ────────────────────────────────────────────────────────────────────────────
# Helper: mult/extreme sweep
# ────────────────────────────────────────────────────────────────────────────
run_mult() {
    local csv="$1"
    local script="$2"
    local -n _layers=$3
    local workdir="$4"

    echo "timestamp,layer,count,factor,accuracy" > "$csv"
    for LAYER in "${_layers[@]}"; do
        for COUNT in "${COUNTS[@]}"; do
            for FACTOR in "${FACTORS[@]}"; do
                local TS=$(date +%Y-%m-%d_%H:%M:%S)
                echo "[$TS] Layer=$LAYER Count=$COUNT Factor=$FACTOR" | tee -a "$GLOBAL_LOG"
                local OUT
                OUT=$(cd "$workdir" && python "$script" --layer "$LAYER" --count "$COUNT" --factor "$FACTOR" 2>&1 | tee -a "$GLOBAL_LOG") || true
                local ACC
                ACC=$(echo "$OUT" | grep "Top-1 Accuracy:" | awk '{print $3}' | tr -d '%')
                [ -z "$ACC" ] && ACC=$(echo "$OUT" | grep "^mAP50:" | awk '{print $2}')
                [ -z "$ACC" ] && ACC="-1"
                echo "$TS,$LAYER,$COUNT,$FACTOR,$ACC" >> "$csv"
            done
        done
    done
    echo "  -> $csv"
}

# ────────────────────────────────────────────────────────────────────────────
# Helper: bit-flip sweep
# ────────────────────────────────────────────────────────────────────────────
run_bit() {
    local csv="$1"
    local script="$2"
    local -n _layers=$3
    local workdir="$4"

    echo "timestamp,layer,count,accuracy" > "$csv"
    for LAYER in "${_layers[@]}"; do
        for COUNT in "${BIT_COUNTS[@]}"; do
            local TS=$(date +%Y-%m-%d_%H:%M:%S)
            echo "[$TS] Layer=$LAYER Count=$COUNT" | tee -a "$GLOBAL_LOG"
            local OUT
            OUT=$(cd "$workdir" && python "$script" --layer "$LAYER" --count "$COUNT" 2>&1 | tee -a "$GLOBAL_LOG") || true
            local ACC
            ACC=$(echo "$OUT" | grep "Top-1 Accuracy:" | awk '{print $3}' | tr -d '%')
            [ -z "$ACC" ] && ACC=$(echo "$OUT" | grep "^mAP50:" | awk '{print $2}')
            [ -z "$ACC" ] && ACC="-1"
            echo "$TS,$LAYER,$COUNT,$ACC" >> "$csv"
        done
    done
    echo "  -> $csv"
}

# ============================================================================

echo ""
echo "--- ResNet20 ---"
run_mult "$OUTPUT_DIR/resnet20_pytorchfi_mult.csv"  "$EXAMPLES_DIR/mult_resnet20_pytorchfi.py"  LAYERS_RESNET    "$EXAMPLES_DIR"
run_mult "$OUTPUT_DIR/resnet20_mrfi_mult.csv"       "$EXAMPLES_DIR/mult_resnet20_mrfi.py"       LAYERS_RESNET    "$EXAMPLES_DIR"
run_bit  "$OUTPUT_DIR/resnet20_pytorchfi_bit.csv"   "$EXAMPLES_DIR/bit_resnet20_pytorchfi.py"   LAYERS_RESNET    "$EXAMPLES_DIR"
run_bit  "$OUTPUT_DIR/resnet20_mrfi_bit.csv"        "$EXAMPLES_DIR/bit_resnet20_mrfi.py"        LAYERS_RESNET    "$EXAMPLES_DIR"

echo "--- MobileNet ---"
run_mult "$OUTPUT_DIR/mobilenet_pytorchfi_mult.csv" "$EXAMPLES_DIR/mult_mobilenet_pytorchfi.py" LAYERS_MOBILENET "$EXAMPLES_DIR"
run_mult "$OUTPUT_DIR/mobilenet_mrfi_mult.csv"      "$EXAMPLES_DIR/mult_mobilenet_mrfi.py"      LAYERS_MOBILENET "$EXAMPLES_DIR"
run_bit  "$OUTPUT_DIR/mobilenet_pytorchfi_bit.csv"  "$EXAMPLES_DIR/bit_mobilenet_pytorchfi.py"  LAYERS_MOBILENET "$EXAMPLES_DIR"
run_bit  "$OUTPUT_DIR/mobilenet_mrfi_bit.csv"       "$EXAMPLES_DIR/bit_mobilenet_mrfi.py"       LAYERS_MOBILENET "$EXAMPLES_DIR"

echo "--- ShuffleNet ---"
run_mult "$OUTPUT_DIR/shufflenet_pytorchfi_mult.csv" "$EXAMPLES_DIR/mult_shufflenet_pytorchfi.py" LAYERS_SHUFFLE "$EXAMPLES_DIR"
run_mult "$OUTPUT_DIR/shufflenet_mrfi_mult.csv"      "$EXAMPLES_DIR/mult_shufflenet_mrfi.py"      LAYERS_SHUFFLE "$EXAMPLES_DIR"
run_bit  "$OUTPUT_DIR/shufflenet_pytorchfi_bit.csv"  "$EXAMPLES_DIR/bit_shufflenet_pytorchfi.py"  LAYERS_SHUFFLE "$EXAMPLES_DIR"
run_bit  "$OUTPUT_DIR/shufflenet_mrfi_bit.csv"       "$EXAMPLES_DIR/bit_shufflenet_mrfi.py"       LAYERS_SHUFFLE "$EXAMPLES_DIR"

echo "--- YOLO ---"
run_mult "$OUTPUT_DIR/yolo_pytorchfi_mult.csv"  "$YOLOV9_DIR/mult_yolo_pytorchfi.py" LAYERS_YOLO "$YOLOV9_DIR"
run_mult "$OUTPUT_DIR/yolo_mrfi_mult.csv"       "$YOLOV9_DIR/mult_yolo_mrfi.py"      LAYERS_YOLO "$YOLOV9_DIR"
run_bit  "$OUTPUT_DIR/yolo_pytorchfi_bit.csv"   "$YOLOV9_DIR/bit_yolo_pytorchfi.py"  LAYERS_YOLO "$YOLOV9_DIR"
run_bit  "$OUTPUT_DIR/yolo_mrfi_bit.csv"        "$YOLOV9_DIR/bit_yolo_mrfi.py"       LAYERS_YOLO "$YOLOV9_DIR"

echo ""
echo "========================================================================"
echo "PyTorchFI & MRFI sweeps complete. Results in: $OUTPUT_DIR"
echo "========================================================================"
