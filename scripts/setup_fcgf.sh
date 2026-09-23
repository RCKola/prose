#!/usr/bin/env bash
# CPU-only setup for the `registration.corr_extractor=fcgf` backend:
# builds MinkowskiEngine without CUDA and fetches the FCGF 3DMatch weights.
#
#     bash scripts/setup_fcgf.sh
#
# For a CUDA build, follow the upstream instructions instead — see
# https://github.com/NVIDIA/MinkowskiEngine and https://github.com/chrischoy/FCGF
#
# Why this is not just `pip install MinkowskiEngine`: MinkowskiEngine 0.5.4
# (Feb 2021) predates most of the current toolchain, so a modern environment
# trips over several things that this script works around:
#
#   1. BLAS is autodetected through `numpy.distutils`, which calls
#      `CCompiler(None, dry_run, force)`. Modern setuptools replaced that class,
#      so it dies with
#          TypeError: Compiler.__init__() takes from 1 to 3 positional arguments
#      Passing --blas=<name> explicitly skips the autodetection branch.
#   2. pip runs PEP 517 metadata generation *before* the build and
#      --config-settings never reaches that phase, so the error above fires
#      anyway. We build a wheel with setup.py directly and pip-install that.
#   3. The build needs cblas.h, which the conda openblas package provides.
#   4. The ABCs were removed from `collections` in Python 3.10, breaking five
#      modules at import time.
#   5. FCGF's own `model/simpleunet.py` carries a `future_fstrings` encoding
#      cookie and needs that shim to import.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ME_VERSION="0.5.4"
WEIGHTS_URL="https://huggingface.co/chrischoy/FCGF/resolve/main/2019-08-19_06-17-41.pth"
WEIGHTS_DIR="$ROOT/weights/fcgf"
WEIGHTS_PATH="$WEIGHTS_DIR/fcgf_3dmatch.pth"

# --- 0. Locate the active Python -------------------------------------------
PY="$(command -v python)"
ENV_PREFIX="$(dirname "$(dirname "$PY")")"
echo "==> Using python: $PY"

# --- 1. FCGF 3DMatch weights ------------------------------------------------
# The URL in FCGF's pre-2026 README (node1.chrischoy.org) is dead; upstream
# rehosted the checkpoints on HuggingFace. This is the ResUNetBN2C 3DMatch
# model: normalized feature, 2.5 cm voxel, 32-dim — what _FCGFRunner expects.
mkdir -p "$WEIGHTS_DIR"
if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "==> Downloading FCGF 3DMatch weights..."
    curl -fsSL -o "$WEIGHTS_PATH" "$WEIGHTS_URL"
else
    echo "==> FCGF weights already present."
fi

# --- 2. Already installed? --------------------------------------------------
if "$PY" -c "import MinkowskiEngine" >/dev/null 2>&1; then
    echo "==> MinkowskiEngine already importable — nothing to build."
    exit 0
fi

# --- 3. BLAS headers --------------------------------------------------------
# The build includes <cblas.h>. conda-forge openblas ships it.
if [ ! -f "$ENV_PREFIX/include/cblas.h" ]; then
    echo "==> cblas.h not found in $ENV_PREFIX/include — installing openblas..."
    conda install -p "$ENV_PREFIX" -c conda-forge openblas libopenblas -y
fi

# --- 4. Fetch the sdist -----------------------------------------------------
SRC_DIR="$(mktemp -d)"
trap 'rm -rf "$SRC_DIR"' EXIT
echo "==> Fetching MinkowskiEngine $ME_VERSION sdist..."
"$PY" -m pip download "MinkowskiEngine==$ME_VERSION" \
    --no-deps --no-binary :all: --no-build-isolation -d "$SRC_DIR" >/dev/null 2>&1 || true
TARBALL="$(find "$SRC_DIR" -name "MinkowskiEngine-*.tar.gz" | head -1)"
if [ -z "$TARBALL" ]; then
    # pip download runs the failing metadata hook; fall back to a direct fetch.
    TARBALL="$SRC_DIR/MinkowskiEngine-$ME_VERSION.tar.gz"
    URL="$("$PY" - "$ME_VERSION" <<'PY'
import json, sys, urllib.request
ver = sys.argv[1]
with urllib.request.urlopen("https://pypi.org/pypi/MinkowskiEngine/json") as r:
    d = json.load(r)
print(next(f["url"] for f in d["releases"][ver] if f["packagetype"] == "sdist"))
PY
)"
    curl -fsSL -o "$TARBALL" "$URL"
fi
tar xzf "$TARBALL" -C "$SRC_DIR"
BUILD_DIR="$SRC_DIR/MinkowskiEngine-$ME_VERSION"

# --- 5. Python 3.10 compatibility ------------------------------------------
# The ABCs moved from `collections` to `collections.abc` in Python 3.3 and the
# aliases were removed in 3.10. ME 0.5.4 targets 3.7, so five modules fail at
# import with: cannot import name 'Sequence' from 'collections'.
echo "==> Rewriting collections ABC imports for Python 3.10+..."
find "$BUILD_DIR/MinkowskiEngine" -name "*.py" -exec sed -i \
    -e 's/^from collections import Sequence, namedtuple$/from collections import namedtuple\nfrom collections.abc import Sequence/' \
    -e 's/^from collections import Sequence$/from collections.abc import Sequence/' \
    {} +

# --- 6. Build the wheel (CPU only) ------------------------------------------
export CXX="${CXX:-g++}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
echo "==> Building MinkowskiEngine (CPU only, MAX_JOBS=$MAX_JOBS)..."
(
    cd "$BUILD_DIR"
    "$PY" setup.py bdist_wheel \
        --cpu_only \
        --blas=openblas \
        --blas_include_dirs="$ENV_PREFIX/include" \
        --blas_library_dirs="$ENV_PREFIX/lib"
)

WHEEL="$(find "$BUILD_DIR/dist" -name "*.whl" | head -1)"
if [ -z "$WHEEL" ]; then
    echo "!! Build produced no wheel." >&2
    exit 1
fi

echo "==> Installing $(basename "$WHEEL")"
"$PY" -m pip install --force-reinstall --no-deps "$WHEEL"

# FCGF's own `model/simpleunet.py` carries a `future_fstrings` encoding cookie,
# so importing ResUNetBN2C needs that shim installed.
"$PY" -m pip install future-fstrings

# --- 7. Verify --------------------------------------------------------------
"$PY" - <<'PY'
import MinkowskiEngine as ME
print(f"    MinkowskiEngine {ME.__version__} OK (CPU only)")
PY

echo
echo "Done. Run the FCGF backend with:"
echo "    python scripts/run_pipeline.py registration.corr_extractor=fcgf"
