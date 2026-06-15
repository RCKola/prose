#!/usr/bin/env python3
"""Preprocess an ADT sequence into the flat layout the pipeline's ADTDataset expects.

This script is intended to run in a dedicated Python 3.11 conda env that has
`projectaria_tools` installed (projectaria_tools does not publish wheels for
Python 3.13, which the main pipeline uses).

    # setup (once):
    conda create -n prose_adt python=3.11 -y
    conda run -n prose_adt pip install projectaria_tools opencv-python

    # run (each sequence):
    conda run -n prose_adt python scripts/preprocess_adt.py \\
        --sequence-dir /data1/nahyuk.lee/etc/adt/Apartment_release_clean_seq133_M1292 \\
        --output /data1/nahyuk.lee/etc/adt_preprocessed \\
        --window-size 6 --stride 5 --frame-interval 0.5

The ADT download ships artifacts under per-artifact subdirectories
(e.g. `ADT_<seq>_main_groundtruth/`, `ADT_<seq>_depth/`). This script first
flattens those into the canonical v2.X layout that `AriaDigitalTwinDataPathsProvider`
expects (via symlinks in a temp dir) before opening the sequence.

Outputs:

    <output>/<sequence>/
        rectified/frame_%06d.png
        depth/frame_%06d.npy
        poses.npy             (N, 4, 4) world→camera
        intrinsics.npy        (3, 3)
        anchors_val.json      sliding-window pairs
        metadata.json         summary of what was written
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List


def _require_projectaria():
    try:
        from projectaria_tools.core import calibration  # noqa: F401
        from projectaria_tools.projects.adt import (  # noqa: F401
            AriaDigitalTwinDataProvider,
            AriaDigitalTwinDataPathsProvider,
        )
    except ImportError as e:
        raise SystemExit(
            "projectaria_tools is required. This script should run in a "
            "Python 3.11 env (main prose env uses 3.13 which has no wheel).\n"
            "    conda create -n prose_adt python=3.11 -y\n"
            "    conda run -n prose_adt pip install projectaria_tools opencv-python\n"
            f"Original: {e}"
        )


def _find_first(patterns: List[Path]) -> Path | None:
    for p in patterns:
        if p.is_file():
            return p
    return None


def flatten_download(raw_dir: Path) -> Path:
    """Symlink the artifacts from the raw download into a v2.X-shaped dir.

    Returns the path to the flat dir (in a tempdir).
    """
    raw_dir = raw_dir.resolve()
    seq_name = raw_dir.name

    flat = Path(tempfile.mkdtemp(prefix=f"adt_flat_{seq_name}_"))

    # Canonical file names → candidate source paths
    def link(canonical: str, candidates: List[Path]) -> bool:
        src = _find_first(candidates)
        if src is None:
            return False
        target = flat / canonical
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(src)
        return True

    # Main VRS (named `video.vrs` in v2.X convention).
    link("video.vrs", list(raw_dir.glob("*main_recording.vrs")))
    # Depth VRS
    link("depth_images.vrs", list((raw_dir.glob("*depth*/depth_images.vrs"))))
    # Main groundtruth CSVs/json go at the top level of the flat dir.
    gt_candidates = list(raw_dir.glob("*main_groundtruth"))
    if gt_candidates:
        gt_dir = gt_candidates[0]
        for p in gt_dir.iterdir():
            tgt = flat / p.name
            if tgt.exists() or tgt.is_symlink():
                tgt.unlink()
            tgt.symlink_to(p.resolve())
    # Optional: MPS trajectories (replaces aria_trajectory.csv when present).
    mps_traj = list(raw_dir.glob("*mps_slam_trajectories"))
    if mps_traj:
        for p in mps_traj[0].iterdir():
            tgt = flat / p.name
            if not (tgt.exists() or tgt.is_symlink()):
                tgt.symlink_to(p.resolve())
    # Optional: MPS calibration.
    mps_calib = list(raw_dir.glob("*mps_slam_calibration"))
    if mps_calib:
        for p in mps_calib[0].iterdir():
            tgt = flat / p.name
            if not (tgt.exists() or tgt.is_symlink()):
                tgt.symlink_to(p.resolve())

    print(f"Flattened {seq_name} → {flat}")
    print(f"  contents: {sorted(p.name for p in flat.iterdir())}")
    return flat


def _build_anchors(frame_ids: List[int], window_size: int, stride: int) -> list:
    pairs = []
    if len(frame_ids) < 2 * window_size:
        return pairs
    for start in range(0, len(frame_ids) - 2 * window_size + 1, stride):
        src_frames = frame_ids[start : start + window_size]
        ref_start = start + stride
        if ref_start + window_size > len(frame_ids):
            break
        ref_frames = frame_ids[ref_start : ref_start + window_size]
        # Poses are in a common world frame, so GT transform is identity.
        pairs.append({
            "src": f"{start:04d}",
            "ref": f"{ref_start:04d}",
            "src_frames": [int(f) for f in src_frames],
            "ref_frames": [int(f) for f in ref_frames],
            "transform": [[1.0,0,0,0],[0,1.0,0,0],[0,0,1.0,0],[0,0,0,1.0]],
            "overlap": 1.0,
        })
    return pairs


def main() -> int:
    _require_projectaria()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True,
                        help="Path to the raw downloaded sequence "
                             "(e.g. /data1/.../Apartment_release_clean_seq133_M1292).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Root output dir. Files land under <output>/<seq_name>/.")
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--frame-interval", type=float, default=0.5,
                        help="Seconds between sampled frames (paper: 0.5).")
    parser.add_argument("--rectified-size", type=int, default=512)
    parser.add_argument("--rectified-focal", type=float, default=280.0)
    parser.add_argument("--stream-id", default="214-1",
                        help="Aria RGB stream id (default matches paper).")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Optional cap on frame count (0 = all).")
    parser.add_argument("--keep-flat", action="store_true",
                        help="Don't delete the flattened symlink tree after run.")
    args = parser.parse_args()

    import cv2
    import numpy as np
    from projectaria_tools.core import calibration
    from projectaria_tools.core.stream_id import StreamId, RecordableTypeId
    from projectaria_tools.projects.adt import (
        AriaDigitalTwinDataProvider,
        AriaDigitalTwinDataPathsProvider,
    )

    def _make_stream_id(spec: str) -> StreamId:
        """Build a StreamId from e.g. '214-1' (RGB camera, instance 1)."""
        type_id_str, instance_str = spec.split("-")
        type_id_num = int(type_id_str)
        instance_id = int(instance_str)
        # Map 214 to RGB_CAMERA_RECORDABLE_CLASS; 211 is SLAM; 231 is eye; etc.
        mapping = {
            214: RecordableTypeId.RGB_CAMERA_RECORDABLE_CLASS,
            1201: RecordableTypeId.SLAM_CAMERA_DATA,
        }
        if type_id_num in mapping:
            return StreamId(mapping[type_id_num], instance_id)
        # Fallback: iterate enum.
        for t in RecordableTypeId.__members__.values():
            if int(t) == type_id_num:
                return StreamId(t, instance_id)
        raise ValueError(f"Unknown stream type id: {type_id_num}")

    flat_dir = flatten_download(args.sequence_dir)
    try:
        paths_provider = AriaDigitalTwinDataPathsProvider(str(flat_dir))
        # ADT v2+: single device per sequence.
        try:
            datapaths = paths_provider.get_datapaths()
        except TypeError:
            # Some versions expect a device index.
            datapaths = paths_provider.get_datapaths(0)

        provider = AriaDigitalTwinDataProvider(datapaths)
        start_ns = provider.get_start_time_ns()
        end_ns = provider.get_end_time_ns()
        step_ns = int(args.frame_interval * 1e9)
        timestamps = list(range(start_ns, end_ns, step_ns))
        if args.max_frames > 0:
            timestamps = timestamps[: args.max_frames]
        print(f"Sampling {len(timestamps)} timestamps @ {args.frame_interval}s, {start_ns}→{end_ns}")

        stream_id = _make_stream_id(args.stream_id)
        src_calib = provider.get_aria_camera_calibration(stream_id)
        dst_calib = calibration.get_linear_camera_calibration(
            args.rectified_size, args.rectified_size, args.rectified_focal, args.stream_id
        )

        out_root = args.output / args.sequence_dir.name
        (out_root / "rectified").mkdir(parents=True, exist_ok=True)
        (out_root / "depth").mkdir(exist_ok=True)

        poses: List[np.ndarray] = []
        frame_ids: List[int] = []
        n_rgb = 0
        n_depth = 0

        for idx, ts in enumerate(timestamps):
            rgb_pack = provider.get_aria_image_by_timestamp_ns(ts, stream_id)
            if rgb_pack is None or not getattr(rgb_pack, "is_valid", lambda: False)():
                continue
            rgb_img = rgb_pack.data().to_numpy_array() if hasattr(rgb_pack, "data") else rgb_pack.to_numpy_array()

            # Rectify.
            rect_rgb = calibration.distort_by_calibration(rgb_img, dst_calib, src_calib)
            cv2.imwrite(
                str(out_root / "rectified" / f"frame_{idx:06d}.png"),
                cv2.cvtColor(rect_rgb, cv2.COLOR_RGB2BGR),
            )
            n_rgb += 1

            # Depth.
            depth_pack = provider.get_depth_image_by_timestamp_ns(ts, stream_id)
            if depth_pack is not None and getattr(depth_pack, "is_valid", lambda: False)():
                depth_img = depth_pack.data().to_numpy_array() if hasattr(depth_pack, "data") else depth_pack.to_numpy_array()
                # depth values are in millimetres by convention — convert to metres.
                depth_m = depth_img.astype(np.float32) / 1000.0
                rect_d = calibration.distort_label_by_calibration(depth_m, dst_calib, src_calib)
                np.save(out_root / "depth" / f"frame_{idx:06d}.npy", rect_d.astype(np.float32))
                n_depth += 1

            # Pose.
            pose_pack = provider.get_aria_3d_pose_by_timestamp_ns(ts)
            if pose_pack is None or not getattr(pose_pack, "is_valid", lambda: False)():
                continue
            pose = pose_pack.data() if hasattr(pose_pack, "data") else pose_pack
            T = np.eye(4, dtype=np.float32)
            # `transform_world_device` (SE3) → 4x4.
            try:
                T_wd = pose.transform_scene_device  # attribute in newer versions
            except AttributeError:
                T_wd = pose.transform_world_device  # older versions
            T[:3, :3] = np.asarray(T_wd.rotation().to_matrix(), dtype=np.float32)
            T[:3, 3] = np.asarray(T_wd.translation(), dtype=np.float32)
            # ADTDataset expects poses as world→camera; pose is device→world, so invert.
            R = T[:3, :3]
            t = T[:3, 3]
            T_wc = np.eye(4, dtype=np.float32)
            T_wc[:3, :3] = R.T
            T_wc[:3, 3] = -R.T @ t
            poses.append(T_wc)
            frame_ids.append(idx)

        if not poses:
            print("ERROR: no valid frames extracted.", file=sys.stderr)
            return 1

        poses_arr = np.stack(poses, axis=0)
        np.save(out_root / "poses.npy", poses_arr)

        K = np.array(
            [[args.rectified_focal, 0.0, args.rectified_size / 2.0],
             [0.0, args.rectified_focal, args.rectified_size / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        np.save(out_root / "intrinsics.npy", K)

        anchors = _build_anchors(frame_ids, args.window_size, args.stride)
        with (out_root / "anchors_val.json").open("w") as f:
            json.dump(anchors, f, indent=2)

        metadata = {
            "sequence": args.sequence_dir.name,
            "rectified_frames": n_rgb,
            "depth_frames": n_depth,
            "valid_poses": len(frame_ids),
            "anchor_pairs": len(anchors),
            "window_size": args.window_size,
            "stride": args.stride,
            "frame_interval_seconds": args.frame_interval,
            "rectified_size": args.rectified_size,
            "rectified_focal": args.rectified_focal,
            "stream_id": args.stream_id,
        }
        with (out_root / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        print(json.dumps(metadata, indent=2))
        return 0
    finally:
        if not args.keep_flat:
            shutil.rmtree(flat_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
