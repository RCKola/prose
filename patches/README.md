# Patches

Small patches applied to upstream third-party code so the pipeline builds on
modern Python / CUDA environments.

## `geotransformer_cpu_build.patch`

Applies three small changes to [qinzheng93/GeoTransformer](https://github.com/qinzheng93/GeoTransformer):

1. **`setup.py`**: `CUDAExtension` → `CppExtension`. GeoTransformer's extension
   sources are all `.cpp` (no `.cu`) — declaring the extension as CUDAExtension
   pulls in CUDA toolkit headers unnecessarily and breaks the build on hosts
   without the full toolkit.
2. **`geotransformer/extensions/common/torch_helper.h`**: removes
   `#include <ATen/cuda/CUDAContext.h>`. Not actually used by the CHECK_CUDA
   macro; its only effect was to force CUDA-toolkit headers into the
   translation unit.
3. **`geotransformer/modules/kpconv/kernel_points.py`**: makes the Open3D
   import optional and adds a `plyfile`-based PLY loader.

Applied automatically by `scripts/setup_geotransformer.sh`.
