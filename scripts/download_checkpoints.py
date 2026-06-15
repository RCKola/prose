#!/usr/bin/env python3
"""Download pretrained checkpoints used by the pipeline.

HuggingFace models (SAM3, Qwen3.6-VL) are downloaded on first `from_pretrained`
call, so they do not need this script — running the pipeline once will fetch
them. VGGT-Omega is gated; download its checkpoint manually (see README).

GeoTransformer ships its 3DMatch-pretrained weights as a GitHub release asset.
We fetch it here into ./weights/geotransformer/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


GEOTRANSFORMER_3DMATCH_URL = (
    "https://github.com/qinzheng93/GeoTransformer/releases/download/v1.0/"
    "geotransformer-3dmatch.pth.tar"
)


def _download_geotransformer(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"Already present: {target}")
        return
    print(f"Downloading {GEOTRANSFORMER_3DMATCH_URL} -> {target}")

    def _report(blocks: int, bs: int, total: int) -> None:
        pct = blocks * bs * 100 / max(total, 1)
        print(f"\r  {pct:5.1f}%", end="", flush=True)

    urlretrieve(GEOTRANSFORMER_3DMATCH_URL, target, _report)
    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geotransformer-out",
        type=Path,
        default=Path("weights/geotransformer/geotransformer-3dmatch.pth.tar"),
    )
    parser.add_argument(
        "--skip-geotransformer",
        action="store_true",
        help="Skip GeoTransformer weight download (assumes already present).",
    )
    args = parser.parse_args()

    if not args.skip_geotransformer:
        _download_geotransformer(args.geotransformer_out)


if __name__ == "__main__":
    main()
