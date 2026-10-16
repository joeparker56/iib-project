"""Prediction helpers for checkpoints from train_geotransolver_surface.py (blade surface + plane model).

Those checkpoints have "kind" == "surface_plane". The model predicts the conserved fields (ro rovx rovr rorvt roe,
the values stored in the tables) at the blade-surface points and at the plane I = plane_i only, so the plot scripts
use `SurfaceModel.predict_case`, which returns actual and predicted fields split into the regions
    "PS" (nI, J)   pressure face, leading to trailing edge
    "SS" (nI, J)   suction face
    "plane" (J, K) the axial plane
and, besides the five conserved fields, the derived vx vr vt p M p0 (tensor_dataset.derive) for both.
"""
import numpy as np
import torch

from geotransolver_torch import GeoTransolver
from tensor_dataset import CONSERVED, derive
from train_geotransolver import LE_I, TE_I
from train_geotransolver_surface import case_points


def is_surface(ckpt):
    return ckpt.get("kind") == "surface_plane"


class SurfaceModel:
    def __init__(self, ckpt):
        a = ckpt["args"]
        self.variables = list(a["variables"])
        if sorted(self.variables) != sorted(CONSERVED):
            raise ValueError(f"expected the conserved variables {CONSERVED}, checkpoint has {self.variables}")
        self.plane_i, self.global_cols, self.norm = a["plane_i"], a["global_cols"], ckpt["norm"]
        self.model = GeoTransolver(
            functional_dim=6, out_dim=len(self.variables), geometry_dim=3, global_dim=len(self.global_cols),
            n_layers=a["n_layers"], n_hidden=a["n_hidden"], n_head=a["n_head"], slice_num=a["slice_num"],
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def predict_points(self, s, row):
        """s: dataset sample built with this model's variables; row: its cases.csv row.
        Returns pred (N, C), actual (N, C) in table units, pts (N, 3) and n_blade (blade rows come first)."""
        nm = self.norm
        pts, nrm, y, nb = case_points(s["coords"], s["fields"], self.plane_i)
        norm_c = lambda c: 2 * (c - nm["c_min"]) / (nm["c_max"] - nm["c_min"]).clamp_min(1e-8) - 1  # noqa: E731
        local = torch.cat([norm_c(pts), nrm], dim=-1)[None]
        glob = torch.tensor([[float(row[c]) for c in self.global_cols]])
        glb = (2 * (glob - nm["g_min"]) / (nm["g_max"] - nm["g_min"]).clamp_min(1e-8) - 1)[:, None, :]
        out = self.model(local, geometry=norm_c(pts[:nb])[None], global_embedding=glb)[0]
        return (out * nm["f_std"] + nm["f_mean"]).numpy(), y.numpy(), pts.numpy(), nb

    def _regions(self, arr, pts, shape, gamma):
        n_i, (_, _, n_j, n_k) = TE_I - LE_I + 1, shape  # shape of the coords: (3, I, J, K)
        n_ps = n_i * n_j
        cuts = {"PS": (slice(0, n_ps), (n_i, n_j)), "SS": (slice(n_ps, 2 * n_ps), (n_i, n_j)),
                "plane": (slice(2 * n_ps, None), (n_j, n_k))}
        per_region = {}
        for name, (sl, shp) in cuts.items():
            f = {v: arr[sl, i].reshape(shp) for i, v in enumerate(self.variables)}
            with np.errstate(all="ignore"):
                f.update(derive(f, pts[sl, 1].reshape(shp), gamma))  # vx vr vt p M p0 (r = column 1 of the points)
            per_region[name] = f
        return {v: {name: per_region[name][v] for name in cuts} for v in per_region["PS"]}

    def predict_case(self, s, row):
        """-> (actual, predicted), each {variable: {"PS": (nI, J), "SS": (nI, J), "plane": (J, K)}}."""
        pred, act, pts, _ = self.predict_points(s, row)
        gamma, shape = float(row["Gamma"]), tuple(s["coords"].shape)
        return self._regions(act, pts, shape, gamma), self._regions(pred, pts, shape, gamma)
