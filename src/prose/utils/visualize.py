"""Per-stage visualization helpers. Write images/PLYs that a human can inspect."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .io import record_io


def _dir_size(path: Path) -> int:
    try:
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError:
        return 0

def load_frame_bgr(path: Path) -> np.ndarray:
    """Load a frame as BGR uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def sample_colors_from_p2p(
    points: np.ndarray,
    point_to_pixels: Sequence[Sequence[dict]],
    frame_paths: Dict[int, Path],
) -> np.ndarray | None:
    """Sample per-point RGB from rectified frames via point_to_pixels.

    Takes the first projection entry per point: (frame_id, pixel_u, pixel_v).
    Returns (N, 3) uint8 RGB, or None if no projections / frames available.
    """
    n = points.shape[0]
    if n == 0 or len(point_to_pixels) != n:
        return None
    colors = np.zeros((n, 3), dtype=np.uint8)
    cache: Dict[int, np.ndarray] = {}
    any_hit = False
    for i, entries in enumerate(point_to_pixels):
        if not entries:
            continue
        e = entries[0]
        fid = int(e.get("frame_id", -1))
        u = int(e.get("pixel_u", -1))
        v = int(e.get("pixel_v", -1))
        if fid < 0 or u < 0 or v < 0:
            continue
        img = cache.get(fid)
        if img is None:
            p = frame_paths.get(fid)
            if p is None or not Path(p).exists():
                cache[fid] = None  # type: ignore[assignment]
                continue
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            img = None if bgr is None else bgr[..., ::-1]  # BGR -> RGB
            cache[fid] = img
        if img is None:
            continue
        H, W = img.shape[:2]
        if 0 <= u < W and 0 <= v < H:
            colors[i] = img[v, u]
            any_hit = True
    return colors if any_hit else None


def save_point_cloud_ply(points: np.ndarray, path: Path, colors: np.ndarray | None = None) -> None:
    """Write (N,3) points to an ASCII PLY so meshlab/Open3D/CloudCompare can open it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = points.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if colors is not None:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header.append("end_header")
    t0 = time.perf_counter()
    with path.open("w") as f:
        f.write("\n".join(header) + "\n")
        for i in range(n):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    record_io("save_ply", path, time.perf_counter() - t0, path.stat().st_size)


def save_mask_overlay(
    image_bgr: np.ndarray,
    masks_by_instance: Dict[int, np.ndarray],
    path: Path,
    alpha: float = 0.5,
) -> None:
    """Save an RGB image with per-instance masks overlaid in different colors + id labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = image_bgr.copy()
    rng = np.random.default_rng(0)
    for iid, mask in masks_by_instance.items():
        color = rng.integers(64, 256, size=3).tolist()
        colored = np.zeros_like(out)
        colored[mask] = color
        out = cv2.addWeighted(out, 1.0, colored, alpha, 0)
        # Draw id text at mask centroid.
        from .mask_ops import mask_centroid
        c = mask_centroid(mask)
        if c is not None:
            cv2.putText(out, str(iid), c, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(path), out)


def save_all_mask_overlays(
    frame_paths: Dict[int, Path],
    per_frame_masks: Dict[int, Dict[int, np.ndarray]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        for fidx, inst_masks in per_frame_masks.items():
            src_path = frame_paths.get(fidx)
            if src_path is None or not Path(src_path).exists():
                continue
            img = load_frame_bgr(src_path)
            save_mask_overlay(img, inst_masks, out_dir / f"frame_{fidx:06d}_masks.png")
    finally:
        record_io("save_overlays", out_dir, time.perf_counter() - t0, _dir_size(out_dir))


# ---------------------------------------------------------------------------
# Pair-level scene snapshots (matplotlib 3D scatter → PNG).
# Headless, no GL; matplotlib is the only dependency.
# ---------------------------------------------------------------------------

def _instance_color(iid: int) -> np.ndarray:
    """Deterministic per-instance RGB in [0,1]. Same iid → same color."""
    rng = np.random.default_rng(int(iid) & 0xFFFFFFFF)
    return rng.uniform(0.25, 1.0, size=3)


def _subsample(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if points.shape[0] <= n:
        return np.arange(points.shape[0])
    return np.random.default_rng(seed).choice(points.shape[0], size=n, replace=False)


def _instance_centroids(
    points: np.ndarray, point_to_inst: np.ndarray
) -> Dict[int, np.ndarray]:
    """Per-instance centroid in the same frame as `points`."""
    out: Dict[int, np.ndarray] = {}
    for iid in np.unique(point_to_inst):
        if iid < 0:
            continue
        out[int(iid)] = points[point_to_inst == iid].mean(axis=0)
    return out


def _draw_cloud(
    ax,
    points: np.ndarray,
    point_to_inst: Optional[np.ndarray],
    *,
    base_rgb: Sequence[float],
    inst_alpha: float = 0.6,
    base_alpha: float = 0.08,
    max_points: int = 30000,
) -> None:
    """Plot a cloud: unassigned points in `base_rgb` (faded) + instance points in their own colors."""
    if points.shape[0] == 0:
        return
    if point_to_inst is None:
        idx = _subsample(points, max_points)
        ax.scatter(
            points[idx, 0], points[idx, 1], points[idx, 2],
            s=0.3, c=[base_rgb], alpha=base_alpha, linewidths=0,
        )
        return
    base_mask = point_to_inst < 0
    inst_mask = ~base_mask
    if base_mask.any():
        bp = points[base_mask]
        bidx = _subsample(bp, max(1, max_points // 2))
        ax.scatter(
            bp[bidx, 0], bp[bidx, 1], bp[bidx, 2],
            s=0.3, c=[base_rgb], alpha=base_alpha, linewidths=0,
        )
    if inst_mask.any():
        ip = points[inst_mask]
        iids = point_to_inst[inst_mask]
        sidx = _subsample(ip, max_points)
        colors = np.stack([_instance_color(int(i)) for i in iids[sidx]])
        ax.scatter(
            ip[sidx, 0], ip[sidx, 1], ip[sidx, 2],
            s=0.6, c=colors, alpha=inst_alpha, linewidths=0,
        )


def _equal_aspect(ax, points_list: List[np.ndarray]) -> None:
    """Set equal-aspect bounds across all panels so scale is comparable."""
    pts = np.concatenate([p for p in points_list if p.shape[0] > 0], axis=0)
    if pts.shape[0] == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2
    span = float((maxs - mins).max()) / 2 + 0.5
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:  # noqa: BLE001 — older matplotlib
        pass


def _three_views(fig, points_list: List[np.ndarray]):
    """Add 3 subplots (top, front, side) and return their Axes3D."""
    axes = []
    for i, (elev, azim, title) in enumerate(
        [(89, -90, "top (XY)"), (10, -90, "front (XZ)"), (10, 0, "side (YZ)")]
    ):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        _equal_aspect(ax, points_list)
        axes.append(ax)
    return axes


def save_pair_snapshots(
    *,
    pair_id: str,
    src_points: np.ndarray,
    ref_points: np.ndarray,
    src_point_to_inst: Optional[np.ndarray],
    ref_point_to_inst: Optional[np.ndarray],
    correspondences: Sequence[Tuple[int, int]],
    est_transform: Optional[np.ndarray],
    gt_transform: Optional[np.ndarray],
    out_dir: Path,
    max_points: int = 30000,
    offset_x: Optional[float] = None,
) -> None:
    """Write two composite PNGs into `out_dir`:

      `{pair_id}_objects_correspondences.png`
        Unaligned view, src and ref shifted apart along x by `offset_x`. Object
        instances colored per-id; matched centroids connected by green lines so
        you can see whether the correspondence set is sensible.

      `{pair_id}_registration.png`
        src transformed by `est_transform` (and optionally `gt_transform`),
        overlaid on ref. Source = blue, ref = orange. No correspondence lines —
        this view is for judging the transform.

    Correspondences arrows degenerate to dots in the registered frame, which is
    why we keep them in the unaligned view only.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    from .pointcloud import apply_transform

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _t_snap0 = time.perf_counter()
    _bytes_before = _dir_size(out_dir)

    if offset_x is None:
        if src_points.shape[0] > 0 and ref_points.shape[0] > 0:
            sx = src_points[:, 0].max() - src_points[:, 0].min()
            rx = ref_points[:, 0].max() - ref_points[:, 0].min()
            offset_x = float(max(sx, rx)) + 1.0
        else:
            offset_x = 5.0

    src_blue = (0.20, 0.45, 0.85)
    ref_orange = (0.95, 0.55, 0.15)

    # ---- Figure 1: objects + correspondences (unaligned, with offset). ----
    src_shifted = src_points.copy()
    if src_shifted.shape[0] > 0:
        src_shifted[:, 0] -= offset_x / 2.0
    ref_shifted = ref_points.copy()
    if ref_shifted.shape[0] > 0:
        ref_shifted[:, 0] += offset_x / 2.0

    fig = plt.figure(figsize=(15, 5), dpi=150)
    fig.suptitle(f"{pair_id} — objects + correspondences (unaligned)", fontsize=11)
    axes = _three_views(fig, [src_shifted, ref_shifted])

    src_centroids = _instance_centroids(src_shifted, src_point_to_inst) if src_point_to_inst is not None else {}
    ref_centroids = _instance_centroids(ref_shifted, ref_point_to_inst) if ref_point_to_inst is not None else {}

    for ax in axes:
        _draw_cloud(ax, src_shifted, src_point_to_inst, base_rgb=src_blue, max_points=max_points)
        _draw_cloud(ax, ref_shifted, ref_point_to_inst, base_rgb=ref_orange, max_points=max_points)
        for sid, rid in correspondences:
            if sid in src_centroids and rid in ref_centroids:
                a, b = src_centroids[sid], ref_centroids[rid]
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                        color=(0.1, 0.7, 0.2), linewidth=0.8, alpha=0.85)
        for iid, c in src_centroids.items():
            ax.scatter([c[0]], [c[1]], [c[2]], s=12, c=[_instance_color(iid)],
                       edgecolors="black", linewidths=0.3)
        for iid, c in ref_centroids.items():
            ax.scatter([c[0]], [c[1]], [c[2]], s=12, c=[_instance_color(iid)],
                       edgecolors="black", linewidths=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"{pair_id}_objects_correspondences.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: registration (aligned overlay). ----
    rows = 1 + (1 if gt_transform is not None else 0)
    fig = plt.figure(figsize=(15, 5 * rows), dpi=150)
    fig.suptitle(f"{pair_id} — registration overlay", fontsize=11)

    def _overlay(row_idx: int, T: Optional[np.ndarray], label: str):
        if T is None or src_points.shape[0] == 0:
            src_T = src_points
        else:
            src_T = apply_transform(src_points, T)
        for col, (elev, azim, title) in enumerate(
            [(89, -90, "top (XY)"), (10, -90, "front (XZ)"), (10, 0, "side (YZ)")]
        ):
            ax = fig.add_subplot(rows, 3, row_idx * 3 + col + 1, projection="3d")
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{label} — {title}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            _equal_aspect(ax, [src_T, ref_points])
            if src_T.shape[0] > 0:
                idx = _subsample(src_T, max_points, seed=1)
                ax.scatter(src_T[idx, 0], src_T[idx, 1], src_T[idx, 2],
                           s=0.3, c=[src_blue], alpha=0.35, linewidths=0)
            if ref_points.shape[0] > 0:
                idx = _subsample(ref_points, max_points, seed=2)
                ax.scatter(ref_points[idx, 0], ref_points[idx, 1], ref_points[idx, 2],
                           s=0.3, c=[ref_orange], alpha=0.35, linewidths=0)

    _overlay(0, est_transform, "estimated" if est_transform is not None else "estimated (none)")
    if gt_transform is not None:
        _overlay(1, gt_transform, "ground truth")

    fig.tight_layout()
    fig.savefig(out_dir / f"{pair_id}_registration.png", bbox_inches="tight")
    plt.close(fig)
    record_io(
        "save_pair_snapshots",
        out_dir,
        time.perf_counter() - _t_snap0,
        max(0, _dir_size(out_dir) - _bytes_before),
    )
