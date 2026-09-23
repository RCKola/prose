#!/usr/bin/env python3
"""Rotate preprocessed ADT artifacts 90° CW so downstream stages see upright frames.

Aria's RGB stream is temple-mounted (portrait orientation); preprocessing
leaves frames sideways (ceiling on the left edge, floor on the right).
This breaks depth estimation, hurts segmentation grounding, and confuses
the VLM. This script rotates everything 90° CW and stores the results
under ``*_rot/`` directories alongside the originals.

Outputs (alongside the existing dirs):
    <seq>/rectified_rot/frame_%06d.png    np.rot90(img, k=-1)
    <seq>/depth_rot/frame_%06d.npy        np.rot90(depth, k=-1)
    <seq>/poses_rot.npy                   R_z(camera-frame) @ w2c
    <seq>/intrinsics_rot.npy              K (unchanged for symmetric K)

Pose math: rotating the image 90° CW maps original camera coords
(x, y, z) to new coords (-y, x, z). So R_new_old is
    [[ 0, -1, 0],
     [ 1,  0, 0],
     [ 0,  0, 1]]
and new_w2c = (block_diag(R_new_old, 1)) @ old_w2c (rotation applies
to both R and t since t lives in camera coords for w2c).

Intrinsics: for a symmetric K (fx=fy, cx=cy=N/2), K is invariant
under 90° rotation. Asserted before save.

Usage:
    python preprocessing/adt/rotate_adt_artifacts.py \\
        --seq-dir sample_data/adt_preprocessed/Apartment_release_clean_seq133_M1292
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


R_NEW_OLD = np.array(
    [[0.0, -1.0, 0.0],
     [1.0,  0.0, 0.0],
     [0.0,  0.0, 1.0]], dtype=np.float32,
)
T_NEW_OLD = np.eye(4, dtype=np.float32)
T_NEW_OLD[:3, :3] = R_NEW_OLD


def rotate_rgb_dir(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("frame_*.png")):
        arr = np.asarray(Image.open(p).convert("RGB"))
        rot = np.rot90(arr, k=-1).copy()
        Image.fromarray(rot).save(dst / p.name)
        n += 1
    return n


def rotate_depth_dir(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("frame_*.npy")):
        d = np.load(p)
        rot = np.rot90(d, k=-1).copy()
        np.save(dst / p.name, rot.astype(np.float32))
        n += 1
    return n


def rotate_poses(src: Path, dst: Path) -> int:
    poses = np.load(src)  # (N, 4, 4) w2c
    rotated = np.einsum("ij,njk->nik", T_NEW_OLD, poses).astype(np.float32)
    np.save(dst, rotated)
    return len(rotated)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-dir", required=True,
                    help="Preprocessed ADT sequence directory "
                         "(e.g. sample_data/adt_preprocessed/Apartment_release_clean_seq133_M1292).")
    args = ap.parse_args()
    seq = Path(args.seq_dir)

    if not seq.is_dir():
        print(f"ERROR: {seq} is not a directory.", file=sys.stderr)
        return 1

    n_rgb = rotate_rgb_dir(seq / "rectified", seq / "rectified_rot")
    print(f"Rotated RGB: {n_rgb}")

    if (seq / "depth").exists():
        n_d = rotate_depth_dir(seq / "depth", seq / "depth_rot")
        print(f"Rotated depth: {n_d}")

    n_p = rotate_poses(seq / "poses.npy", seq / "poses_rot.npy")
    print(f"Rotated poses: {n_p}")

    K = np.load(seq / "intrinsics.npy")
    assert abs(K[0, 0] - K[1, 1]) < 1e-3, "fx != fy; K not invariant under 90° rotation"
    assert abs(K[0, 2] - K[1, 2]) < 1e-3, "cx != cy; K not invariant"
    np.save(seq / "intrinsics_rot.npy", K.astype(np.float32))
    print("K (unchanged): saved to intrinsics_rot.npy")


if __name__ == "__main__":
    sys.exit(main() or 0)
