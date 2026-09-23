"""VGGT-Omega wrapper.

VGGT-Omega is a feed-forward depth + camera estimator from Meta AI. Source:
    https://github.com/facebookresearch/vggt-omega

Selected by the Stage 1 (geometry) Hydra config group (`geometry=vggt_omega`).

The HuggingFace model (`facebook/VGGT-Omega`) is gated, so checkpoints must
be downloaded once and pointed at via `checkpoint_path`. Live HF resolution
is intentionally not supported (CSCS compute nodes are offline).

If the `vggt_omega` package is not importable, the wrapper raises an
ImportError; callers should catch it and fall back to GT point clouds.

GT-pose alignment
-----------------
VGGT-Omega predicts poses in an arbitrary world frame at an arbitrary
(non-metric) scale. To make its output drop-in compatible with the DA3
path — which lands points in the 3RScan world frame at metric scale — the
wrapper optionally runs Umeyama Sim(3) alignment of predicted vs GT camera
trajectories:
    * scale the predicted depth by the recovered Umeyama scale,
    * replace predicted extrinsics with the GT extrinsics (3x4 slice).
Disable via `align_to_gt_poses=False` to trust VGGT-Omega's own poses.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from ..utils.gpu import release_gpu
from ..utils.logging import get_logger
from ..utils.profiling import record_model_call

log = get_logger(__name__)


@dataclass
class VGGTOmegaPrediction:
    depth: np.ndarray       # (N, H, W) float32 — at preprocessed image resolution
    conf: np.ndarray        # (N, H, W) float32, ≥ 1.0
    extrinsics: np.ndarray  # (N, 3, 4) float32, world-to-camera
    intrinsics: np.ndarray  # (N, 3, 3) float32 — matched to preprocessed resolution
    is_metric: int = 0      # 1 if Umeyama scaling was applied
    scale_factor: Optional[float] = None  # Umeyama scale (None if alignment skipped)


class VGGTOmegaWrapper:
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        *,
        device: str = "cuda",
        image_resolution: int = 512,
    ) -> None:
        try:
            from vggt_omega.models import VGGTOmega  # type: ignore
            from vggt_omega.utils.load_fn import load_and_preprocess_images  # type: ignore  # noqa: F401
            from vggt_omega.utils.pose_enc import encoding_to_camera  # type: ignore  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "VGGT-Omega requires the `vggt_omega` package. Install with:\n"
                "  git submodule update --init third_party/vggt-omega\n"
                "  # and ensure third_party/vggt-omega is on PYTHONPATH\n"
                f"Original import error: {e}"
            )

        import torch

        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"VGGT-Omega checkpoint not found: {ckpt}. Download once from "
                "https://huggingface.co/facebook/VGGT-Omega (gated) and point "
                "geometry.checkpoint_path at the local .pt file."
            )

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        log.info("Loading VGGT-Omega checkpoint %s on %s", ckpt, self.device)
        model = VGGTOmega().to(self.device).eval()
        state = torch.load(str(ckpt), map_location="cpu")
        if isinstance(state, dict) and "model" in state and "state_dict" not in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        self.model = model
        self.model_id = ckpt.name
        self.image_resolution = int(image_resolution)

    def close(self) -> None:
        model = getattr(self, "model", None)
        self.model = None
        release_gpu(model)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def infer(
        self,
        images: Sequence[Union[str, Path, np.ndarray]],
        *,
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        align_to_gt_poses: bool = True,
    ) -> VGGTOmegaPrediction:
        """Run multi-view inference on a list of images.

        Args:
            images: list of image paths (np.ndarray inputs are not supported by
                VGGT-Omega's `load_and_preprocess_images`; pass paths).
            extrinsics: (N, 4, 4) world-to-camera GT poses, for Umeyama
                alignment when align_to_gt_poses=True. Ignored otherwise.
            intrinsics: unused (kept for API parity with DA3Wrapper). VGGT-Omega
                returns its own intrinsics matched to the preprocessed depth.
            align_to_gt_poses: if True AND extrinsics provided, align predicted
                poses+depth to GT via Umeyama Sim(3) (mirrors DA3's
                `align_to_input_ext_scale=True`).
        """
        from vggt_omega.utils.load_fn import load_and_preprocess_images  # type: ignore
        from vggt_omega.utils.pose_enc import encoding_to_camera  # type: ignore
        import torch

        img_paths = []
        for p in images:
            if not isinstance(p, (str, Path)):
                raise TypeError(
                    "VGGT-Omega `infer` expects image PATHS, got "
                    f"{type(p).__name__}. (DA3's np.ndarray path is not supported.)"
                )
            img_paths.append(str(p))

        imgs = load_and_preprocess_images(
            img_paths, image_resolution=self.image_resolution
        ).to(self.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            preds = self.model(imgs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        record_model_call(
            self.model_id, "infer",
            time.perf_counter() - t0, n_inputs=len(img_paths),
        )

        # predictions are batched: (B=1, N, ...). Strip the batch dim and
        # squeeze the trailing depth channel.
        depth = preds["depth"][0]                       # (N, H, W, 1) or (N, H, W)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        conf = preds["depth_conf"][0]                   # (N, H, W)
        img_hw = preds["images"].shape[-2:]             # (H, W) at preprocessed res

        ext_pred, intr_pred = encoding_to_camera(preds["pose_enc"], img_hw)
        ext_pred = ext_pred[0].float().cpu().numpy()    # (N, 3, 4) w2c
        intr_pred = intr_pred[0].float().cpu().numpy()  # (N, 3, 3)
        depth_np = depth.float().cpu().numpy().astype(np.float32)
        conf_np = conf.float().cpu().numpy().astype(np.float32)

        scale: Optional[float] = None
        ext_out = ext_pred
        if align_to_gt_poses and extrinsics is not None:
            from ..utils.pointcloud import align_poses_umeyama

            # Pad pred to (N,4,4) for the aligner.
            n = ext_pred.shape[0]
            ext_pred_4 = np.zeros((n, 4, 4), dtype=np.float64)
            ext_pred_4[:, :3, :] = ext_pred
            ext_pred_4[:, 3, 3] = 1.0
            ext_gt_4 = extrinsics.astype(np.float64)
            _, _, s = align_poses_umeyama(ext_gt_4, ext_pred_4)
            scale = float(s)
            depth_np = depth_np * scale
            # Replace pred extrinsics with GT (sliced to 3,4) — same as DA3.
            ext_out = extrinsics[:, :3, :].astype(np.float32)

        return VGGTOmegaPrediction(
            depth=depth_np,
            conf=conf_np,
            extrinsics=ext_out.astype(np.float32),
            intrinsics=intr_pred.astype(np.float32),
            is_metric=1 if scale is not None else 0,
            scale_factor=scale,
        )
