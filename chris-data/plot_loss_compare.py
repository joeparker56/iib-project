"""Total pressure loss at the surface model's plane (I=75): actual vs volume model vs surface model, and against the
dataset's own loss, one point per case.

Four values per case:
    actual @ plane    from the CFD fields at the plane
    volume @ plane    from the whole-mesh model's predicted fields at the plane
    surface @ plane   from the surface model's predicted fields at the plane
    dataset loss      pp_p0_loss from cases.csv, which is defined at the product plane (I=84), further downstream
Two definitions are computed at the plane, mass-averaged with weight rho*vx:
    Yp = (p01 - p02) / (p02 - p2)        and        1 - p02/p01
p01 is the case's inlet total pressure (pp_p0_inlet, not predicted by either model). Panels: Yp parity, 1 - p02/p01
parity (both against the actual value at the plane), and the plane loss from all three sources against pp_p0_loss.

    python plot_loss_compare.py --volume_ckpt best.pt --surface_ckpt best_surface.pt --split val --cache val_plots/.cache
    python plot_loss_compare.py --volume_ckpt best.pt --surface_ckpt best_surface.pt --split all --csv loss75.csv --out loss75.png
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from plot_compare_models import VolumeModel, split_label  # noqa: E402
from plot_loss_coefficient import NEEDED, losses  # noqa: E402
from surface_model import SurfaceModel, is_surface  # noqa: E402
from tensor_dataset import VARIABLES, CascadeDataset  # noqa: E402


def stats(pred, act):
    e = pred - act
    return (f"RMSE {np.sqrt((e**2).mean()):.2e}, MAE {np.abs(e).mean():.2e}, bias {e.mean():+.2e}, "
            f"R2 {1 - (e**2).sum() / ((act - act.mean()) ** 2).sum():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume_ckpt", required=True, help="checkpoint from train_geotransolver.py")
    ap.add_argument("--surface_ckpt", required=True, help="checkpoint from train_geotransolver_surface.py")
    ap.add_argument("--data", default=str(ROOT / "grid125_R23"))
    ap.add_argument("--split", choices=["val", "train", "all"], default="val",
                    help="val/train: cases in that split for both models; all: every case in the dataset")
    ap.add_argument("--cache", default=None, help="directory to cache the volume model's whole-mesh predictions")
    ap.add_argument("--out", default=None, help="save the figure to this file instead of showing it")
    ap.add_argument("--csv", default=None, help="also write per-case values to this CSV")
    a = ap.parse_args()

    vck = torch.load(a.volume_ckpt, map_location="cpu", weights_only=False)
    sck = torch.load(a.surface_ckpt, map_location="cpu", weights_only=False)
    if is_surface(vck) or not is_surface(sck):
        sys.exit("--volume_ckpt must come from train_geotransolver.py and --surface_ckpt from train_geotransolver_surface.py")
    vol, sm = VolumeModel(a.volume_ckpt, vck, a.cache), SurfaceModel(sck)
    plane_i = sm.plane_i
    if any(v not in vol.variables for v in NEEDED) and not {"ro", "vx", "vr", "vt", "p"} <= set(vol.variables):
        sys.exit(f"the volume model needs {NEEDED}, it has {vol.variables}")

    ds_act = CascadeDataset(a.data, variables=VARIABLES)
    ds_vol = CascadeDataset(a.data, variables=vol.variables)
    ds_srf = CascadeDataset(a.data, variables=sm.variables)
    n = len(ds_act)
    label = lambda i: (split_label(vck, n, i), split_label(sck, n, i))  # noqa: E731
    chosen = [i for i in range(n) if a.split == "all" or label(i) == (a.split, a.split)]
    if not chosen:
        sys.exit(f"no case is in the {a.split} split for both models")

    recs = []
    for k, i in enumerate(chosen, 1):
        row, case = ds_act.rows[i], ds_act.rows[i]["case"]
        gamma, p01 = float(row["Gamma"]), float(row["pp_p0_inlet"])
        s = ds_act[i]
        rt, r = s["coords"][2, plane_i].numpy(), s["coords"][1, plane_i].numpy()
        act_f = {v: s["fields"][VARIABLES.index(v), plane_i].numpy() for v in NEEDED}
        vol_f = {v: f[plane_i] for v, f in vol.fields(ds_vol[i], case, gamma).items() if v in NEEDED}
        _, srf_all = sm.predict_case(ds_srf[i], ds_srf.rows[i])
        srf_f = {v: srf_all[v]["plane"] for v in NEEDED}
        out = {}
        for name, f in (("actual", act_f), ("volume", vol_f), ("surface", srf_f)):
            out[f"yp_{name}"], out[f"loss_{name}"] = losses(f, gamma, p01, rt, r)
        recs.append(dict(case=case, volume_split=label(i)[0], surface_split=label(i)[1], plane=plane_i, **out,
                         dataset_loss=float(row["pp_p0_loss"])))
        print(f"\r{k}/{len(chosen)} cases", end="", flush=True)
    print()

    arr = lambda key: np.array([x[key] for x in recs])  # noqa: E731
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0]))
            w.writeheader()
            w.writerows(recs)
        print("wrote", a.csv)

    print(f"\nplane I={plane_i}, {len(recs)} cases ({a.split})")
    for title, key in (("Yp", "yp"), ("1 - p02/p01", "loss")):
        print(f"{title}: actual mean {arr(key + '_actual').mean():.5f}")
        for name in ("volume", "surface"):
            print(f"  {name:8s} vs actual at the plane: {stats(arr(f'{key}_{name}'), arr(key + '_actual'))}")
    d = arr("dataset_loss")
    print(f"dataset pp_p0_loss (I=84) mean {d.mean():.5f}; actual loss at the plane (I={plane_i}) mean "
          f"{arr('loss_actual').mean():.5f}, so the loss grows by {d.mean() - arr('loss_actual').mean():+.5f} downstream")
    for name in ("actual", "volume", "surface"):
        print(f"  {name:8s} loss at the plane vs dataset pp_p0_loss: {stats(arr('loss_' + name), d)}")

    colours = {"volume": "tab:blue", "surface": "tab:orange", "actual": "0.35"}
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    for ax, (title, key) in zip(axs[:2], (("Yp = (p01 - p02) / (p02 - p2)", "yp"), ("1 - p02 / p01", "loss"))):
        act = arr(key + "_actual")
        allv = np.concatenate([act, arr(key + "_volume"), arr(key + "_surface")])
        lo, hi = allv.min(), allv.max()
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="perfect")
        for name in ("volume", "surface"):
            ax.scatter(act, arr(f"{key}_{name}"), s=22, c=colours[name], alpha=0.8,
                       label=f"{name} model, R$^2$ {1 - ((arr(f'{key}_{name}') - act) ** 2).sum() / ((act - act.mean()) ** 2).sum():.2f}")
        ax.set_xlabel(f"actual at plane I={plane_i}")
        ax.set_ylabel("predicted")
        ax.set_title(title + f" at I={plane_i}")
        ax.set_aspect("equal")
        ax.legend(loc="upper left")
    ax = axs[2]
    allv = np.concatenate([d, arr("loss_actual"), arr("loss_volume"), arr("loss_surface")])
    lo, hi = allv.min(), allv.max()
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="y = x")
    for name in ("actual", "volume", "surface"):
        ax.scatter(d, arr("loss_" + name), s=22, c=colours[name], alpha=0.8, label=f"{name} at I={plane_i}")
    ax.set_xlabel("dataset pp_p0_loss (actual, I=84)")
    ax.set_ylabel(f"1 - p02/p01 at I={plane_i}")
    ax.set_title("loss at the plane vs the dataset's loss")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.suptitle(f"Total pressure loss at plane I={plane_i}: actual vs volume model vs surface model "
                 f"({a.split} cases, {len(recs)} scored)")
    if a.out:
        fig.savefig(a.out, dpi=150)
        print("saved", a.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
