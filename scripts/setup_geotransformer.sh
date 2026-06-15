#!/usr/bin/env bash
# Initialize the GeoTransformer submodule, apply local patches, build the
# C++ extension, and download the 3DMatch weights.
#
# Run once after cloning:
#     bash scripts/setup_geotransformer.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBMODULE="$ROOT/third_party/GeoTransformer"
PATCH="$ROOT/patches/geotransformer_cpu_build.patch"
WEIGHTS_DIR="$ROOT/weights/geotransformer"
WEIGHTS_URL="https://github.com/qinzheng93/GeoTransformer/releases/download/1.0.0/geotransformer-3dmatch.pth.tar"

cd "$ROOT"

# --- 1. Initialise submodule ----------------------------------------------
if [ ! -f "$SUBMODULE/setup.py" ]; then
    echo "==> Initialising third_party/GeoTransformer submodule..."
    git submodule update --init third_party/GeoTransformer
fi

# --- 2. Apply our CPU-build patch -----------------------------------------
cd "$SUBMODULE"
if git apply --check "$PATCH" 2>/dev/null; then
    echo "==> Applying CPU-build patch..."
    git apply "$PATCH"
else
    # Either already applied or conflicts with local edits — skip quietly.
    if git apply --reverse --check "$PATCH" 2>/dev/null; then
        echo "==> Patch already applied, skipping."
    else
        echo "!! Patch does not apply cleanly. Review $PATCH against your submodule state."
    fi
fi

# --- 3. Build the C++ extension -------------------------------------------
echo "==> Building GeoTransformer C++ extension (in-place)..."
python setup.py build_ext --inplace

# --- 4. Download 3DMatch weights ------------------------------------------
mkdir -p "$WEIGHTS_DIR"
if [ ! -f "$WEIGHTS_DIR/geotransformer-3dmatch.pth.tar" ]; then
    echo "==> Downloading GeoTransformer 3DMatch weights..."
    curl -sL -o "$WEIGHTS_DIR/geotransformer-3dmatch.pth.tar" "$WEIGHTS_URL"
else
    echo "==> GeoTransformer weights already present."
fi

echo
echo "Done. Add the submodule to PYTHONPATH before running the pipeline:"
echo "    export PYTHONPATH=\"$SUBMODULE:\$PYTHONPATH\""
echo
echo "Also set LD_LIBRARY_PATH so torch's shared libs resolve:"
echo "    export LD_LIBRARY_PATH=\"\$(python -c 'import torch, os; print(os.path.dirname(torch.__file__)+\"/lib\")'):\$LD_LIBRARY_PATH\""
