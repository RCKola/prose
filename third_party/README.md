# Third-party submodules

PROSE depends on three external repositories, tracked as git submodules in
[`.gitmodules`](../.gitmodules). The code itself is not vendored — fetch it with:

```bash
git submodule update --init --recursive
```

| Path | Repo | Used by |
|:--|:--|:--|
| `vggt-omega/` | [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega) | geometry stage (`use_gt_pointclouds=false`) |
| `GeoTransformer/` | [qinzheng93/GeoTransformer](https://github.com/qinzheng93/GeoTransformer) | registration (`corr_extractor=geotransformer`, default) |
| `FCGF/` | [chrischoy/FCGF](https://github.com/chrischoy/FCGF) | registration (`corr_extractor=fcgf`) |

After init:

- **GeoTransformer**: run `bash scripts/setup_geotransformer.sh` to apply the
  CPU-build patch, build the C++ extension, and download the 3DMatch weights.
  Add it to `PYTHONPATH` before running the pipeline.
- **vggt-omega**: ensure it is on `PYTHONPATH`; download the gated checkpoint
  (see the main README) and set `VGGT_OMEGA_CKPT`.
- **FCGF**: needs MinkowskiEngine; place the 3DMatch checkpoint at
  `weights/fcgf/fcgf_3dmatch.pth`. Only required for the FCGF backend.

SAM 3 and Qwen3.6-27B are pulled from HuggingFace at runtime — no submodule.
