"""FCGF (chrischoy) — learned 32-dim sparse-conv descriptor for Stage 6.

Drop-in sibling of `_GeoTransformerRunner.run(sp, rp) -> {src, ref, scores}`.
Pipeline: voxel quantize -> sparse ResUNetBN2C forward -> 32-dim normalized
feature per point -> mutual NN in feature space -> p2p correspondences.

`scores` is `cos(src_feat, ref_feat)` ∈ [-1, 1] so the existing `top-K by
scores` cap keeps the highest-similarity matches.

Loads weights from `cfg_registration.corr_extractor_checkpoint` (default
`weights/fcgf/fcgf_3dmatch.pth`, the chrischoy ResUNetBN2C/3DMatch ckpt).

TODO: port to WarpConvNet (upstream marks the MinkowskiEngine paths legacy).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np


_FCGF_PATH = Path(__file__).resolve().parents[3] / "third_party" / "FCGF"


def _ensure_fcgf_on_path() -> None:
    if str(_FCGF_PATH) not in sys.path:
        sys.path.insert(0, str(_FCGF_PATH))


class _FCGFRunner:
    """FCGF + mutual-NN correspondence extractor.

    The pretrained checkpoint shipped with FCGF (`2019-08-19_06-17-41.pth`,
    'normalized feature, 3DMatch, 2.5 cm voxel, 32-dim') expects
    `voxel_size=0.025` — keep that default.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        max_points: int = 10000,
        voxel_size: float = 0.025,
        feat_dim: int = 32,
    ):
        _ensure_fcgf_on_path()
        from model.resunet import ResUNetBN2C  # type: ignore
        import torch

        import MinkowskiEngine as ME  # noqa: F401

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.voxel_size = float(voxel_size)
        self.max_points = int(max_points)
        self.feat_dim = int(feat_dim)

        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        # `normalize_feature=True` matches the 3DMatch FCGF training recipe.
        self.model = ResUNetBN2C(1, self.feat_dim, normalize_feature=True, conv1_kernel_size=7, D=3)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()

    def _featurize(self, points: np.ndarray):
        """Voxel quantize → sparse forward → (Nq, feat) features + (Nq, 3) coords."""
        import MinkowskiEngine as ME
        torch = self.torch

        pts = points.astype(np.float32)
        # Quantize to integer voxel grid (FCGF training uses ME.utils.sparse_quantize).
        coords = np.floor(pts / self.voxel_size).astype(np.int32)
        _, sel = ME.utils.sparse_quantize(coords, return_index=True)
        if len(sel) == 0:
            return None
        coords_q = coords[sel]
        feats = np.ones((len(sel), 1), dtype=np.float32)
        xyz_q = pts[sel]

        coords_q_t = torch.from_numpy(coords_q).int()
        # Prepend batch index (B=1) — MinkowskiEngine requires (B, x, y, z).
        batch = torch.zeros((coords_q_t.shape[0], 1), dtype=torch.int32)
        coords_b = torch.cat([batch, coords_q_t], dim=1).to(self.device)
        feats_t = torch.from_numpy(feats).to(self.device)

        st = ME.SparseTensor(features=feats_t, coordinates=coords_b)
        with torch.inference_mode():
            out = self.model(st)
        feat = out.F.detach().cpu().numpy().astype(np.float32)
        return xyz_q, feat

    def run(self, src_pts: np.ndarray, ref_pts: np.ndarray) -> Optional[dict]:
        if src_pts.shape[0] < 4 or ref_pts.shape[0] < 4:
            return None
        try:
            src_xyz, src_feat = self._featurize(src_pts)
            ref_xyz, ref_feat = self._featurize(ref_pts)
        except Exception:  # noqa: BLE001
            return None

        if src_feat is None or ref_feat is None:
            return None
        if src_feat.shape[0] < 4 or ref_feat.shape[0] < 4:
            return None

        # cap each side to max_points by random downsample (descriptors don't
        # know about coverage, so a uniform random keeps things tractable).
        rng = np.random.default_rng(0)
        if src_feat.shape[0] > self.max_points:
            idx = rng.choice(src_feat.shape[0], self.max_points, replace=False)
            src_feat = src_feat[idx]; src_xyz = src_xyz[idx]
        if ref_feat.shape[0] > self.max_points:
            idx = rng.choice(ref_feat.shape[0], self.max_points, replace=False)
            ref_feat = ref_feat[idx]; ref_xyz = ref_xyz[idx]

        from scipy.spatial import cKDTree

        # FCGF features are L2-normalized → KDTree (L2) gives same ranking
        # as cosine similarity. Use Euclidean for cleaner score semantics.
        ref_tree = cKDTree(ref_feat)
        src_tree = cKDTree(src_feat)

        d_sr, idx_sr = ref_tree.query(src_feat, k=1, workers=-1)
        nearest_ref = idx_sr if idx_sr.ndim == 1 else idx_sr[:, 0]
        d_rs, idx_rs = src_tree.query(ref_feat[nearest_ref], k=1, workers=-1)
        nearest_back = idx_rs if idx_rs.ndim == 1 else idx_rs[:, 0]
        mutual = nearest_back == np.arange(len(src_feat))
        if not mutual.any():
            return None

        sc = src_xyz[mutual].astype(np.float32)
        rc = ref_xyz[nearest_ref[mutual]].astype(np.float32)
        d_keep = d_sr[mutual] if d_sr.ndim == 1 else d_sr[mutual, 0]
        # cos(a,b) = 1 - 0.5 * ||a-b||^2 for unit-norm features.
        scores = (1.0 - 0.5 * d_keep ** 2).astype(np.float32)

        return {"src": sc, "ref": rc, "scores": scores}
