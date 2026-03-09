#!/bin/bash
# Figure 5 — Baseline sweep: PyTorchFI + TensorDynamic.
#
# Baseline: count=500, factor=1000 (1 run per layer/model)
# Sweep:    count x factor = {10000,50000,100000} x {1000,10000,100000} = 9 runs per layer/model
#
# For TensorDynamic: count=NUM_THREADS_RECORD, factor=CORRUPT_MULT_F32 (INJECTION_MODE=2)
# Same 1st, 3rd, and 5th conv layers as figure4 sweep (KERNEL_LIST=1 3 5).
# Can be run from any directory.

set -o pipefail

FIGURE5_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSORDYNAMIC_DIR="$(dirname "$FIGURE5_DIR")"
EXAMPLES_DIR="$TENSORDYNAMIC_DIR/examples"
YOLOV9_DIR="$EXAMPLES_DIR/yolo/yolov9"
BOUNDARY_DIR="$EXAMPLES_DIR/boundary"
TOOL_SO="$TENSORDYNAMIC_DIR/tensor_dynamic/tensor_dynamic.so"

OUTPUT_DIR="$FIGURE5_DIR/results_baseline"
mkdir -p "$OUTPUT_DIR"
GLOBAL_LOG="$OUTPUT_DIR/sweep.log"

echo "========================================================================"
echo "Figure 5 — Baseline Sweep (PyTorchFI + TensorDynamic)"
echo "TensorDynamic: $TENSORDYNAMIC_DIR"
echo "Examples:      $EXAMPLES_DIR"
echo "Output:        $OUTPUT_DIR"
echo "========================================================================"

# ────────────────────────────────────────────────────────────────────────────
# Build tensor_dynamic.so with Figure 5 limits (max = 110000)
# ────────────────────────────────────────────────────────────────────────────
echo "--- Building tensor_dynamic (MAX=110000) ---"
(cd "$TENSORDYNAMIC_DIR/tensor_dynamic" && make clean && make \
    MAX_TARGETS=110000 \
    MAX_RECORDED_THREADS=110000 \
    MAX_INJECTION_RECORDS=110000 \
    MAX_INJECTED_TRACKING=110000)
echo "--- Build complete ---"
echo ""

# ────────────────────────────────────────────────────────────────────────────
# Parameter sets
# ────────────────────────────────────────────────────────────────────────────

# Baseline: single fixed point
BASELINE_COUNT=500
BASELINE_FACTOR=1000

# Sweep: 3 x 3 = 9 combinations
SWEEP_COUNTS=(10000 50000 100000)
SWEEP_FACTORS=(1000 10000 100000)

# 1st, 3rd, and 5th conv layers (index 0, 2, and 4 from each model's layer list)
LAYERS_RESNET=("conv1" "layer1.0.conv2" "layer1.1.conv2")
LAYERS_MOBILENET=("features.0.0" "features.1.conv.1" "features.2.conv.1.0")
LAYERS_SHUFFLE=("conv1.0" "stage2.0.branch1.2" "stage2.0.branch2.3")
LAYERS_YOLO=("model.0.conv" "model.2.cv1.conv" "model.2.cv2.0.m.0.cv1.conv")

# ────────────────────────────────────────────────────────────────────────────
# Helper: run one mult injection and append result to CSV
# ────────────────────────────────────────────────────────────────────────────
run_one() {
    local csv="$1"
    local script="$2"
    local workdir="$3"
    local layer="$4"
    local count="$5"
    local factor="$6"
    local tag="$7"   # "baseline" or "sweep"
    local outdir="$8"

    local safe_layer
    safe_layer="$(echo "$layer" | tr './' '__')"
    local outfile="$outdir/${safe_layer}_${count}_${factor}.txt"

    local TS
    TS=$(date +%Y-%m-%d_%H:%M:%S)
    echo "[$TS] [$tag] Layer=$layer Count=$count Factor=$factor" | tee -a "$GLOBAL_LOG"
    local OUT ACC
    for attempt in $(seq 1 $((MAX_RETRIES + 1))); do
        OUT=$(cd "$workdir" && python "$script" --layer "$layer" --count "$count" --factor "$factor" 2>&1 | tee -a "$GLOBAL_LOG" | tee "$outfile") && break
        if [ "$attempt" -le "$MAX_RETRIES" ]; then
            echo "[RETRY $attempt/$MAX_RETRIES] $tag Layer=$layer Count=$count Factor=$factor" | tee -a "$GLOBAL_LOG"
        else
            echo "[FAILED] $tag Layer=$layer Count=$count Factor=$factor" | tee -a "$GLOBAL_LOG"
        fi
    done
    ACC=$(echo "$OUT" | grep -oP "(?<=Top-1 Accuracy: )[\d.]+")
    [ -z "$ACC" ] && ACC=$(echo "$OUT" | grep -oP "(?<=^mAP50: )[\d.]+")
    [ -z "$ACC" ] && ACC="-1"
    echo "$TS,$tag,$layer,$count,$factor,$ACC" >> "$csv"
}

# ────────────────────────────────────────────────────────────────────────────
# Run all layers for one model (baseline + 9-point sweep)
# ────────────────────────────────────────────────────────────────────────────
run_model() {
    local csv="$1"
    local script="$2"
    local -n _layers=$3
    local workdir="$4"
    local outdir="${csv%.csv}"
    mkdir -p "$outdir"

    echo "timestamp,tag,layer,count,factor,accuracy" > "$csv"

    for LAYER in "${_layers[@]}"; do
        # Baseline only
        run_one "$csv" "$script" "$workdir" "$LAYER" "$BASELINE_COUNT" "$BASELINE_FACTOR" "baseline" "$outdir"
    done

    echo "  -> $csv"
}

# TensorDynamic: kernel positions 1, 3, and 5 = 1st, 3rd, and 5th conv layers
KERNEL_LIST=(1 3 5)
INS_LIST=(1)
MAX_RETRIES=10

# ────────────────────────────────────────────────────────────────────────────
# Helper: run one TensorDynamic mult injection and append result to CSV
# ────────────────────────────────────────────────────────────────────────────
run_td_one() {
    local csv="$1"
    local eval_script="$2"
    local boundary="$3"
    local nt="$4"    # NUM_THREADS_RECORD (≈ count)
    local cl="$5"    # CORRUPT_MULT_F32   (≈ factor)
    local k="$6"     # TARGET_KERNEL_POS
    local ins="$7"   # TARGET_INSTR_LIST
    local tag="$8"   # "baseline" or "sweep"
    local outdir="$9"
    local outfile="$outdir/nt${nt}_cl${cl}_k${k}_i${ins}.txt"
    # CUDA_LAUNCH_BLOCKING=1 \

    local TS
    TS=$(date +%Y-%m-%d_%H:%M:%S)
    echo "[$TS] [TensorDynamic $tag] NT=$nt CL=$cl K=$k I=$ins" | tee -a "$GLOBAL_LOG"
    local OUT ACC
    local workdir="$(dirname "$eval_script")"
    for attempt in $(seq 1 $((MAX_RETRIES + 1))); do
        OUT=$((cd "$workdir" && \
            NUM_THREADS_RECORD=$nt \
            INJECTION_MODE=2 \
            CORRUPT_MULT_F32=$cl \
            TARGET_KERNEL_POS=$k \
            TARGET_INSTR_LIST="$ins" \
            BOUNDARY_PATH="$boundary" \
            LD_PRELOAD="$TOOL_SO" \
            python3 "$eval_script") 2>&1 | tee -a "$GLOBAL_LOG" | tee "$outfile") && break
        if [ "$attempt" -le "$MAX_RETRIES" ]; then
            echo "[RETRY $attempt/$MAX_RETRIES] TensorDynamic $tag NT=$nt CL=$cl K=$k I=$ins" | tee -a "$GLOBAL_LOG"
        else
            echo "[FAILED] TensorDynamic $tag NT=$nt CL=$cl K=$k I=$ins" | tee -a "$GLOBAL_LOG"
        fi
    done
    ACC=$(echo "$OUT" | grep -oP "(?<=Top-1 Accuracy: )[\d.]+")
    [ -z "$ACC" ] && ACC=$(echo "$OUT" | grep -oP "(?<=^mAP50: )[\d.]+")
    [ -z "$ACC" ] && ACC="-1"
    echo "$TS,$tag,kernel$k,$nt,$cl,$ACC" >> "$csv"
}

run_tensordynamic() {
    local csv="$1"
    local eval_script="$2"
    local boundary="$3"
    local outdir="${csv%.csv}"
    mkdir -p "$outdir"

    echo "timestamp,tag,kernel,count,factor,accuracy" > "$csv"

    for K in "${KERNEL_LIST[@]}"; do
        for INS in "${INS_LIST[@]}"; do
            # 9-point sweep only
            for NT in "${SWEEP_COUNTS[@]}"; do
                for CL in "${SWEEP_FACTORS[@]}"; do
                    run_td_one "$csv" "$eval_script" "$boundary" "$NT" "$CL" "$K" "$INS" "sweep" "$outdir"
                done
            done
        done
    done
    echo "  -> $csv"
}

# ============================================================================
echo ""
echo "=== [1/2] PyTorchFI ==="

echo "--- ResNet20 ---"
run_model "$OUTPUT_DIR/resnet20_pytorchfi_mult.csv" "$EXAMPLES_DIR/mult_resnet20_pytorchfi.py" LAYERS_RESNET "$EXAMPLES_DIR"

echo "--- MobileNet ---"
run_model "$OUTPUT_DIR/mobilenet_pytorchfi_mult.csv" "$EXAMPLES_DIR/mult_mobilenet_pytorchfi.py" LAYERS_MOBILENET "$EXAMPLES_DIR"

echo "--- ShuffleNet ---"
run_model "$OUTPUT_DIR/shufflenet_pytorchfi_mult.csv" "$EXAMPLES_DIR/mult_shufflenet_pytorchfi.py" LAYERS_SHUFFLE "$EXAMPLES_DIR"

echo "--- YOLO ---"
run_model "$OUTPUT_DIR/yolo_pytorchfi_mult.csv" "$YOLOV9_DIR/mult_yolo_pytorchfi.py" LAYERS_YOLO "$YOLOV9_DIR"

# ============================================================================
echo ""
echo "=== [2/2] TensorDynamic ==="

echo "--- ResNet20 ---"
run_tensordynamic "$OUTPUT_DIR/resnet20_tensordynamic_mult.csv" \
    "$EXAMPLES_DIR/eval_resnet20.py" "$BOUNDARY_DIR/boundary_resnet20.txt"

echo "--- MobileNet ---"
run_tensordynamic "$OUTPUT_DIR/mobilenet_tensordynamic_mult.csv" \
    "$EXAMPLES_DIR/eval_mobilenet.py" "$BOUNDARY_DIR/boundary_mobilenet.txt"

echo "--- ShuffleNet ---"
run_tensordynamic "$OUTPUT_DIR/shufflenet_tensordynamic_mult.csv" \
    "$EXAMPLES_DIR/eval_shufflenet.py" "$BOUNDARY_DIR/boundary_shufflenet.txt"

echo "--- YOLO ---"
run_tensordynamic "$OUTPUT_DIR/yolo_tensordynamic_mult.csv" \
    "$YOLOV9_DIR/eval_yolo.py" "$BOUNDARY_DIR/boundary_yolo.txt"

# ============================================================================
echo ""
echo "========================================================================"
echo "All sweeps complete. Results in: $OUTPUT_DIR"
echo "========================================================================"
