#!/bin/bash
# Setup script for COCO val2017 dataset for YOLOv9 evaluation
# Run from the TensorDynamic directory: bash setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COCO_DIR="$SCRIPT_DIR/examples/yolo/datasets/coco"

echo "========================================"
echo "COCO val2017 Dataset Setup"
echo "========================================"
echo "Target directory: $COCO_DIR"
echo ""

YOLO_DIR="$SCRIPT_DIR/examples/yolo"
YOLOV9_DIR="$YOLO_DIR/yolov9"
WEIGHTS_DIR="$YOLO_DIR/weights"

# Create directory structure
mkdir -p "$COCO_DIR/images"
mkdir -p "$WEIGHTS_DIR"

# Clone YOLOv9
if [ -d "$YOLOV9_DIR/.git" ]; then
    echo "[1/6] YOLOv9 repo already cloned, skipping."
else
    echo "[1/6] Cloning YOLOv9..."
    # Remove partial clone if any
    rm -rf "$YOLOV9_DIR"
    git clone https://github.com/WongKinYiu/yolov9.git "$YOLOV9_DIR"
fi

# Patch models/experimental.py for PyTorch >= 2.6 (weights_only now defaults to True)
EXPERIMENTAL="$YOLOV9_DIR/models/experimental.py"
if grep -q "weights_only=False" "$EXPERIMENTAL"; then
    echo "[2/6] PyTorch 2.6 patch already applied, skipping."
else
    echo "[2/6] Patching models/experimental.py for PyTorch >= 2.6..."
    sed -i 's/torch\.load(attempt_download(w), map_location='"'"'cpu'"'"')/torch.load(attempt_download(w), map_location='"'"'cpu'"'"', weights_only=False)/' "$EXPERIMENTAL"
fi

# Copy custom scripts into yolov9/
echo "[3/6] Copying custom scripts into yolov9/..."
for f in mult_yolo_mrfi.py mult_yolo_pytorchfi.py profile_yolo.py bit_yolo_mrfi.py bit_yolo_pytorchfi.py eval_yolo.py; do
    cp "$SCRIPT_DIR/examples/$f" "$YOLOV9_DIR/$f"
done

# Download val2017 images (~1GB)
if [ -d "$COCO_DIR/images/val2017" ]; then
    echo "[4/6] val2017 images already exist, skipping download."
else
    echo "[4/6] Downloading val2017 images (~1GB)..."
    wget -q --show-progress -P "$COCO_DIR/images" \
        http://images.cocodataset.org/zips/val2017.zip
    echo "Extracting val2017 images..."
    unzip -q "$COCO_DIR/images/val2017.zip" -d "$COCO_DIR/images"
    rm "$COCO_DIR/images/val2017.zip"
fi

# Download YOLO-format labels
if [ -d "$COCO_DIR/labels/val2017" ]; then
    echo "[5/6] YOLO labels already exist, skipping download."
else
    echo "[5/6] Downloading YOLO-format labels..."
    wget --progress=bar:force -O "$COCO_DIR/coco2017labels.zip" \
        https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip
    echo "Extracting val2017 labels only..."
    unzip "$COCO_DIR/coco2017labels.zip" "coco/labels/val2017/*" -d "$COCO_DIR"
    rm "$COCO_DIR/coco2017labels.zip"
    # The zip extracts into a nested coco/ subdirectory; move contents up
    if [ -d "$COCO_DIR/coco/labels" ]; then
        mv "$COCO_DIR/coco/labels" "$COCO_DIR/labels"
        rm -rf "$COCO_DIR/coco"
    fi
fi

# Download YOLOv9-S weights
if [ -f "$WEIGHTS_DIR/yolov9-c-converted.pt" ]; then
    echo "[6/6] YOLOv9-C weights already exist, skipping download."
else
    echo "[6/6] Downloading YOLOv9-C weights..."
    wget --progress=bar:force -O "$WEIGHTS_DIR/yolov9-c-converted.pt" \
        https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt
fi

# Generate val2017.txt image list
echo "Generating val2017.txt..."
find "$COCO_DIR/images/val2017" -name "*.jpg" | sort | while read f; do
    echo "./images/val2017/$(basename "$f")"
done > "$COCO_DIR/val2017.txt"
echo "  $(wc -l < "$COCO_DIR/val2017.txt") images listed."

echo ""
echo "========================================"
echo "Setup complete."
echo "Dataset is ready at: $COCO_DIR"
echo ""
echo "Expected structure:"
echo "  $COCO_DIR/"
echo "  ├── images/val2017/   (5000 images)"
echo "  ├── labels/val2017/   (YOLO labels)"
echo "  └── val2017.txt       (image list)"
echo "  $WEIGHTS_DIR/"
echo "  └── yolov9-c-converted.pt"
echo ""
echo "Run evaluation from:"
echo "  cd $SCRIPT_DIR/examples/yolo/yolov9"
echo "  python eval_yolo.py"
echo "========================================"
