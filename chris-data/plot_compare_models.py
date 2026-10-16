"""Actual vs the whole-mesh (volume) model vs the blade-surface model, on the blade faces and the plane.

Rows: pressure face, suction face, plane. Columns: actual, volume model, surface model, and the two errors.
Both models are scored on the same points: the blade faces (LE..TE) and the plane at the surface model's plane_i.

    python plot_compare_models.py --volume_ckpt best.pt --surface_ckpt best_surface.pt
    python plot_compare_models.py --volume_ckpt best.pt --surface_ckpt best_surface.pt --case g134 --var M --out cmp.png
    python plot_compare_models.py --volume_ckpt best.pt --surface_ckpt best_surface.pt --split all --out_dir compare \\
        --cache pred_cache
--var is any of ro vx vr vt p M p0 (rovx rovr rorvt roe too, if the volume model was trained on them). With the
default settings both training scripts draw the same validation split; each case is labelled train/val per model.
The volume model predicts the whole mesh (slow on CPU), so use --cache to reuse those predictions.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from geotransolver_torch import GeoTransolver  # noqa: E402
from predict_cache import cached_predict_field  # noqa: E402
from surface_model import SurfaceModel, is_surface  # noqa: E402
from tensor_dataset import CONSERVED, VARIABLES, CascadeDataset, derive  # noqa: E402
from train_geotransolver import LE_I, TE_I, blade_surface, build_model  # noqa: E402

REGIONS = (("PS", "pressure side"), ("SS", "suction side"), ("plane", "plane"))


def split_label(ckpt, n_cases, case_idx):
    """'train' / 'val' / 'not used' for a dataset case index under this checkpoint's split."""
    picks = ckpt.get("picks") or torch.linspace(0, n_cases - 1, len(ckpt["train_idx"]) + len(ckpt["val_idx"])).round().long().tolist()
    if case_idx in [picks[i] for i in ckpt["val_idx"]]:
        return "val"
    return "train" if case_idx in [picks[i] for i in ckpt["train_idx"]] else "not used"


class VolumeModel:
    """The whole-mesh model (train_geotransolver.py checkpoints)."""

    def __init__(self, path, ckpt, cache):
        self.path, self.ckpt, self.cache = path, ckpt, cache
        a = ckpt["args"]
        self.variables, self.n_points, self.nm = a["variables"], a["n_points"], ckpt["norm"]
        self.model = build_model(a, len(self.variables))
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def fields(self, s, case, gamma):
        """-> {variable: (I, J, K)} in table units, including M and p0 when they can be derived."""
        nm, c = self.nm, s["coords"]
        norm = lambda x: 2 * (x - nm["c_min"]) / (nm["c_max"] - nm["c_min"]).clamp_min(1e-8) - 1  # noqa: E731
        g = (2 * (s["params"] - nm["p_min"]) / (nm["p_max"] - nm["p_min"]).clamp_min(1e-8) - 1)[None, None]
        full = cached_predict_field(self.cache, self.path, case, self.model, norm(c.flatten(1).T), g,
                                    norm(blade_surface(c))[None], self.n_points)
        full = (full * nm["f_std"] + nm["f_mean"]).reshape(*c.shape[1:], -1).numpy()
        f = {v: full[..., i] for i, v in enumerate(self.variables)}
        with np.errstate(all="ignore"):
            if set(CONSERVED) <= set(f):
                f.update(derive(f, c[1].numpy(), gamma))
            elif {"ro", "vx", "vr", "vt", "p"} <= set(f):
                v2 = f["vx"] ** 2 + f["vr"] ** 2 + f["vt"] ** 2
                f["M"] = np.sqrt(v2 / (gamma * f["p"] / f["ro"]))
                f["p0"] = f["p"] * (1 + 0.5 * (gamma - 1) * f["M"] ** 2) ** (gamma / (gamma - 1))
        return f


def to_regions(f, plane_i):
    """{variable: (I, J, K)} -> {variable: {"PS": (nI, J), "SS": (nI, J), "plane": (J, K)}}"""
    sl = slice(LE_I, TE_I + 1)
    return {v: {"PS": a[sl, :, 0], "SS": a[sl, :, -1], "plane": a[plane_i]} for v, a in f.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume_ckpt", required=True, help="checkpoint from train_geotransolver.py")
    ap.add_argument("--surface_ckpt", required=True, help="checkpoint from train_geotransolver_surface.py")
    ap.add_argument("--data", default=str(ROOT / "grid125_R23"))
    ap.add_argument("--case", nargs="+", default=None, help="case ids (default: first validation case of both)")
    ap.add_argument("--split", choices=["val", "train", "all"], default=None,
                    help="every case in this split: val/train = cases that are in that split for both models, "
                         "all = all cases in the dataset")
    ap.add_argument("--all_val", action="store_true", help="same as --split val")
    ap.add_argument("--var", default="p", help="ro vx vr vt p M p0 (and the conserved fields if both models have them)")
    ap.add_argument("--cache", default=None, help="directory to cache the volume model's whole-mesh predictions")
    ap.add_argument("--out", default=None, help="save a single case to this file instead of showing it")
    ap.add_argument("--out_dir", default=None, help="save <case>.png for each case here (needed for several cases)")
    ap.add_argument("--skip_existing", action="store_true")
    a = ap.parse_args()

    vck = torch.load(a.volume_ckpt, map_location="cpu", weights_only=False)
    sck = torch.load(a.surface_ckpt, map_location="cpu", weights_only=False)
    if is_surface(vck) or not is_surface(sck):
        sys.exit("--volume_ckpt must come from train_geotransolver.py and --surface_ckpt from train_geotransolver_surface.py")
    vol = VolumeModel(a.volume_ckpt, vck, a.cache)
    sm = SurfaceModel(sck)
    plane_i = sm.plane_i

    ds_act = CascadeDataset(a.data, variables=VARIABLES)  # actual fields incl. the derived channels
    ds_vol = CascadeDataset(a.data, variables=vol.variables)  # (the volume model only needs coords and params)
    ds_srf = CascadeDataset(a.data, variables=sm.variables)  # the surface model's fields are in its own order
    ids = [r["case"] for r in ds_act.rows]
    n = len(ids)

    both_val = [i for i in range(n) if split_label(vck, n, i) == "val" and split_label(sck, n, i) == "val"]
    cases = list(a.case or [])
    split = a.split or ("val" if a.all_val else None)
    if split:
        both_train = [i for i in range(n) if split_label(vck, n, i) == "train" and split_label(sck, n, i) == "train"]
        chosen = {"val": both_val, "train": both_train, "all": list(range(n))}[split]
        cases += [ids[i] for i in chosen if ids[i] not in cases]
    if not cases:
        if not both_val:
            sys.exit("no case is a validation case for both models; pass --case")
        cases = [ids[both_val[0]]]
    unknown = [c for c in cases if c not in ids]
    if unknown:
        sys.exit(f"unknown case ids: {unknown}")
    if len(cases) > 1 and not a.out_dir:
        sys.exit("several cases need --out_dir (one PNG per case)")
    if a.out and len(cases) > 1:
        sys.exit("--out is for a single case; use --out_dir")

    rows_out = []
    for k, case in enumerate(cases, 1):
        i = ids.index(case)
        target = Path(a.out_dir) / f"{case}.png" if a.out_dir else None
        if target and a.skip_existing and target.exists():
            continue
        row = ds_act.rows[i]
        gamma = float(row["Gamma"])
        s_act = ds_act[i]
        act = to_regions({v: s_act["fields"][j].numpy() for j, v in enumerate(VARIABLES)}, plane_i)
        vol_r = to_regions(vol.fields(ds_vol[i], case, gamma), plane_i)
        _, srf_r = sm.predict_case(ds_srf[i], ds_srf.rows[i])
        for name, d in (("actual", act), ("volume model", vol_r), ("surface model", srf_r)):
            if a.var not in d:
                sys.exit(f"{a.var!r} not available for the {name}; it has {sorted(d)}")
        v_lab, s_lab = split_label(vck, n, i), split_label(sck, n, i)
        print(f"case {case} (volume model: {v_lab}, surface model: {s_lab}); plane I={plane_i}", flush=True)

        c = s_act["coords"].numpy()
        sl = slice(LE_I, TE_I + 1)
        xy = {"PS": (c[0, sl, :, 0], c[1, sl, :, 0]), "SS": (c[0, sl, :, -1], c[1, sl, :, -1]),
              "plane": (c[2, plane_i], c[1, plane_i])}  # (horizontal, vertical) coordinates of each region
        xlab = {"PS": "x (axial)", "SS": "x (axial)", "plane": "rt (pitchwise)"}

        fig, axs = plt.subplots(3, 5, figsize=(25, 13), constrained_layout=True)
        stats = {}
        for r, (reg, reg_name) in enumerate(REGIONS):
            A, V, S = act[a.var][reg], vol_r[a.var][reg], srf_r[a.var][reg]
            ev, es = V - A, S - A
            vals = np.concatenate([x[np.isfinite(x)].ravel() for x in (A, V, S)])
            vnorm = Normalize(vals.min(), vals.max())
            emax = max(np.nanmax(np.abs(ev)), np.nanmax(np.abs(es))) or 1.0
            enorm = TwoSlopeNorm(0, -emax, emax)
            cols = [("Actual", A, vnorm, "viridis"), ("Volume model", V, vnorm, "viridis"),
                    ("Surface model", S, vnorm, "viridis"), ("Volume error", ev, enorm, "RdBu_r"),
                    ("Surface error", es, enorm, "RdBu_r")]
            for ci, (title, z, norm, cmap) in enumerate(cols):
                ax = axs[r, ci]
                ax.pcolormesh(*xy[reg], z, norm=norm, cmap=cmap, shading="gouraud")
                ax.set_title(f"{title} - {reg_name}")
                ax.set_xlabel(xlab[reg])
                if ci == 0:
                    ax.set_ylabel("r (hub to midspan)")
                if reg == "plane":
                    ax.set_aspect("equal")
            fig.colorbar(cm.ScalarMappable(vnorm, "viridis"), ax=axs[r, :3], shrink=0.9, label=a.var)
            fig.colorbar(cm.ScalarMappable(enorm, "RdBu_r"), ax=axs[r, 3:], shrink=0.9, label=f"{a.var} error")
            nrm = lambda x: float(np.sqrt(np.nanmean(x**2)))  # noqa: E731
            stats[reg] = {"volume": nrm(ev) / nrm(A), "surface": nrm(es) / nrm(A),
                          "volume_dev": nrm(ev - np.nanmean(ev)) / nrm(A - np.nanmean(A)),
                          "surface_dev": nrm(es - np.nanmean(es)) / nrm(A - np.nanmean(A))}
            print(f"  {reg_name:13s} rel L2 error  volume {stats[reg]['volume']:.4f}  surface {stats[reg]['surface']:.4f}"
                  f"  | of variation about the mean  volume {stats[reg]['volume_dev']:.3f}  surface {stats[reg]['surface_dev']:.3f}")
            rows_out.append([case, reg, a.var, stats[reg]["volume"], stats[reg]["surface"],
                             stats[reg]["volume_dev"], stats[reg]["surface_dev"]])
        fig.suptitle(f"{a.var}, case {case}: actual vs volume model vs surface model "
                     f"(volume model: {v_lab}, surface model: {s_lab}; plane I={plane_i})", fontsize=15)
        if target:
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=120)
            print(f"[{k}/{len(cases)}] saved {target}", flush=True)
        elif a.out:
            fig.savefig(a.out, dpi=120)
            print("saved", a.out)
        else:
            plt.show()
        plt.close(fig)
    if a.out_dir and rows_out:
        with open(Path(a.out_dir) / f"errors_{a.var}.csv", "a", newline="") as fh:
            csv.writer(fh).writerows(rows_out)  # case, region, var, rel L2 (volume, surface), rel L2 of variation


if __name__ == "__main__":
    main()
