#!/bin/bash
# Figure 4 — TensorDynamic (NVBit) fault injection sweep for all four models.
# Can be run from any directory.

set -o pipefail

FIGURE4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSORDYNAMIC_DIR="$(dirname "$FIGURE4_DIR")"
EXAMPLES_DIR="$TENSORDYNAMIC_DIR/examples"
YOLOV9_DIR="$EXAMPLES_DIR/yolo/yolov9"
BOUNDARY_DIR="$EXAMPLES_DIR/boundary"

# Allow OUTPUT_DIR to be passed in (used when called from sweep_all.sh)
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$FIGURE4_DIR/results_tensordynamic"
    mkdir -p "$OUTPUT_DIR"
fi
GLOBAL_LOG="$OUTPUT_DIR/sweep.log"

echo "========================================================================"
echo "TensorDynamic (NVBit) Sweep — all models"
echo "TensorDynamic: $TENSORDYNAMIC_DIR"
echo "Examples:      $EXAMPLES_DIR"
echo "Output:        $OUTPUT_DIR"
echo "========================================================================"

# ────────────────────────────────────────────────────────────────────────────
# Build tensor_dynamic.so with Figure 4 limits (max = 10000)
# ────────────────────────────────────────────────────────────────────────────
echo "--- Building tensor_dynamic (MAX=10000) ---"
(cd "$TENSORDYNAMIC_DIR/tensor_dynamic" && make clean && make \
    MAX_TARGETS=10000 \
    MAX_RECORDED_THREADS=10000 \
    MAX_INJECTION_RECORDS=10000 \
    MAX_INJECTED_TRACKING=10000)
echo "--- Build complete ---"
echo ""

# ────────────────────────────────────────────────────────────────────────────
# Parameter sets
# ────────────────────────────────────────────────────────────────────────────
TOOL_SO="$TENSORDYNAMIC_DIR/tensor_dynamic/tensor_dynamic.so"
THREADS_LIST=(10 50 100 500 1000)
CORRUPT_LIST=(10 50 100 1000)
KERNEL_LIST=(1 3 5)
INS_LIST=(1)
MAX_RETRIES=10

# ────────────────────────────────────────────────────────────────────────────
# Helper: TensorDynamic sweep (bit or mult mode)
# ────────────────────────────────────────────────────────────────────────────
run_tensordynamic() {
    local label="$1"       # e.g. "resnet20"
    local boundary="$2"    # absolute path to boundary file
    local eval_script="$3" # absolute path to eval script
    local mode="$4"        # "mult" or "bit"
    local workdir="$(dirname "$eval_script")"

    local results_dir="$OUTPUT_DIR/tensordynamic_${label}_${mode}"
    mkdir -p "$results_dir"

    if [ "$mode" = "bit" ]; then
        for NT in "${THREADS_LIST[@]}"; do
            for K in "${KERNEL_LIST[@]}"; do
                for I in "${INS_LIST[@]}"; do
                    echo "[TensorDynamic bit] $label NT=$NT K=$K I=$I" | tee -a "$GLOBAL_LOG"
                    local outfile="$results_dir/bit_${NT}_${K}_${I}.txt"
                    for attempt in $(seq 1 $((MAX_RETRIES + 1))); do
                        (cd "$workdir" && \
                            NUM_THREADS_RECORD=$NT \
                            INJECTION_MODE=0 \
                            BIT_POSITION=30 \
                            TARGET_KERNEL_POS=$K \
                            TARGET_INSTR_LIST="$I" \
                            BOUNDARY_PATH="$boundary" \
                            LD_PRELOAD="$TOOL_SO" \
                            python3 "$eval_script") > "$outfile" 2>&1 && break
                        if [ "$attempt" -le "$MAX_RETRIES" ]; then
                            echo "[RETRY $attempt/$MAX_RETRIES] bit NT=$NT K=$K I=$I" | tee -a "$GLOBAL_LOG"
                        else
                            echo "[FAILED] bit NT=$NT K=$K I=$I" | tee -a "$GLOBAL_LOG"
                        fi
                    done
                done
            done
        done
    else
        for NT in "${THREADS_LIST[@]}"; do
            for CL in "${CORRUPT_LIST[@]}"; do
                for K in "${KERNEL_LIST[@]}"; do
                    for I in "${INS_LIST[@]}"; do
                        echo "[TensorDynamic mult] $label NT=$NT CL=$CL K=$K I=$I" | tee -a "$GLOBAL_LOG"
                        local outfile="$results_dir/mult_${NT}_${CL}_${K}_${I}.txt"
                        for attempt in $(seq 1 $((MAX_RETRIES + 1))); do
                            (cd "$workdir" && \
                                NUM_THREADS_RECORD=$NT \
                                INJECTION_MODE=2 \
                                CORRUPT_MULT_F32=$CL \
                                TARGET_KERNEL_POS=$K \
                                TARGET_INSTR_LIST="$I" \
                                BOUNDARY_PATH="$boundary" \
                                LD_PRELOAD="$TOOL_SO" \
                                python3 "$eval_script") > "$outfile" 2>&1 && break
                            if [ "$attempt" -le "$MAX_RETRIES" ]; then
                                echo "[RETRY $attempt/$MAX_RETRIES] mult NT=$NT CL=$CL K=$K I=$I" | tee -a "$GLOBAL_LOG"
                            else
                                echo "[FAILED] mult NT=$NT CL=$CL K=$K I=$I" | tee -a "$GLOBAL_LOG"
                            fi
                        done
                    done
                done
            done
        done
    fi
    echo "  -> $results_dir/"
}

# ============================================================================

echo ""
echo "--- ResNet20 ---"
run_tensordynamic "resnet20"   "$BOUNDARY_DIR/boundary_resnet20.txt"     "$EXAMPLES_DIR/eval_resnet20.py"   "bit"
run_tensordynamic "resnet20"   "$BOUNDARY_DIR/boundary_resnet20.txt"     "$EXAMPLES_DIR/eval_resnet20.py"   "mult"

echo "--- MobileNet ---"
run_tensordynamic "mobilenet"  "$BOUNDARY_DIR/boundary_mobilenet.txt"  "$EXAMPLES_DIR/eval_mobilenet.py"  "bit"
run_tensordynamic "mobilenet"  "$BOUNDARY_DIR/boundary_mobilenet.txt"  "$EXAMPLES_DIR/eval_mobilenet.py"  "mult"

echo "--- ShuffleNet ---"
run_tensordynamic "shufflenet" "$BOUNDARY_DIR/boundary_shufflenet.txt" "$EXAMPLES_DIR/eval_shufflenet.py" "bit"
run_tensordynamic "shufflenet" "$BOUNDARY_DIR/boundary_shufflenet.txt" "$EXAMPLES_DIR/eval_shufflenet.py" "mult"

echo "--- YOLO ---"
run_tensordynamic "yolo"       "$BOUNDARY_DIR/boundary_yolo.txt"       "$YOLOV9_DIR/eval_yolo.py"         "bit"
run_tensordynamic "yolo"       "$BOUNDARY_DIR/boundary_yolo.txt"       "$YOLOV9_DIR/eval_yolo.py"         "mult"

echo ""
echo "========================================================================"
echo "TensorDynamic sweeps complete. Results in: $OUTPUT_DIR"
echo "========================================================================"
