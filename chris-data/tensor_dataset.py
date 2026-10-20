"""PyTorch dataset for the grid125_R23 turbine-cascade flow-field predictions.

Each case becomes a dict of tensors:
    fields : (C, I, J, K) float32   flow variables (see VARIABLES)
    coords : (3, I, J, K) float32   x, r, rt
    params : (3,)         float32   sweep_w_deg, lean_w_deg, theta_unit
    gamma  : ()           float32
    qoi    : (n_qoi,)     float32   product-plane QoIs (pp_*)
    index  : (3,)         int64     (i_sweep, j_lean, k_theta)

Axes are (I axial, J hub->midspan, K pitchwise). The last 2 axial stations
(I = 85, 86) are corrupted by the exit BC (see README) and are dropped by
default, giving I = 85.

Pandas is deliberately not used (the local pandas is incompatible with NumPy 2).
"""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

R_GAS = 287.05
CONSERVED = ("ro", "rovx", "rovr", "rorvt", "roe")
DERIVED = ("vx", "vr", "vt", "p", "M", "p0")
VARIABLES = CONSERVED + DERIVED
N_AXIAL_USABLE = 85  # I = 0..84


def derive(f, r, gamma):
    """Primitive variables from the stored conserved fields (README formulae)."""
    ro = f["ro"]
    vx, vr, vt = f["rovx"] / ro, f["rovr"] / ro, f["rorvt"] / (ro * r)
    v2 = vx**2 + vr**2 + vt**2
    p = (gamma - 1) * (f["roe"] - 0.5 * ro * v2)
    M = np.sqrt(v2 / (gamma * p / ro))
    p0 = p * (1 + 0.5 * (gamma - 1) * M**2) ** (gamma / (gamma - 1))
    return dict(vx=vx, vr=vr, vt=vt, p=p, M=M, p0=p0)


class CascadeDataset(Dataset):
    """Lazily loads one case per __getitem__.

    variables: any subset/order of VARIABLES. Default is the 5 conserved fields.
    n_axial:   number of axial stations kept (default 85; do not exceed it).
    cases:     optional list of case ids (e.g. ["g222"]) to restrict to.
    """

    def __init__(self, root="grid125_R23", variables=CONSERVED, n_axial=N_AXIAL_USABLE, cases=None):
        self.root = Path(root)
        self.variables = tuple(variables)
        unknown = set(self.variables) - set(VARIABLES)
        if unknown:
            raise ValueError(f"unknown variables {sorted(unknown)}; choose from {VARIABLES}")
        self.n_axial = n_axial
        with open(self.root / "cases.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if cases is not None:
            keep = set(cases)
            rows = [r for r in rows if r["case"] in keep]
        self.rows = rows
        self.qoi_names = [k for k in rows[0] if k.startswith("pp_")]
        self._grid_cache = {}

    def __len__(self):
        return len(self.rows)

    def _grid(self, name):
        if name not in self._grid_cache:
            with np.load(self.root / "grids" / name) as g:
                self._grid_cache[name] = np.stack([g["x"], g["r"], g["rt"]])[:, : self.n_axial]
        return self._grid_cache[name]

    def __getitem__(self, idx):
        row = self.rows[idx]
        sl = slice(0, self.n_axial)
        with np.load(self.root / "fields" / f"{row['case']}.npz") as z:
            f = {k: z[k][sl] for k in CONSERVED}
        coords = self._grid(row["grid"])
        gamma = float(row["Gamma"])
        if set(self.variables) & set(DERIVED):
            f.update(derive(f, coords[1], gamma))
        fields = np.stack([f[v] for v in self.variables]).astype(np.float32)
        return {
            "case": row["case"],
            "fields": torch.from_numpy(fields),
            "coords": torch.from_numpy(coords.copy()),
                "params": torch.tensor([float(row[k]) for k in ("sweep_w_deg", "lean_w_deg", "theta_unit")]),
            "gamma": torch.tensor(gamma),
            "qoi": torch.tensor([float(row[k]) for k in self.qoi_names]),
            "index": torch.tensor([int(row[k]) for k in ("i_sweep", "j_lean", "k_theta")]),
        }


if __name__ == "__main__":
    #loads case g222 as an example 
    ds = CascadeDataset(variables=VARIABLES)
    s = ds[ds.rows.index(next(r for r in ds.rows if r["case"] == "g222"))]
    print(len(ds), "cases")
    for k, v in s.items():
        print(k, tuple(v.shape) if hasattr(v, "shape") else v)
    print("finite:", bool(torch.isfinite(s["fields"]).all()))
    for name, ch in zip(ds.variables, s["fields"]):
        print(f"{name:5s} min {ch.min():12.4g} max {ch.max():12.4g}")
