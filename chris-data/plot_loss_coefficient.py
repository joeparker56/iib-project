"""Total pressure loss at the product plane: actual (CFD) vs predicted by the model, one point per case.

Two definitions are computed from the exit-plane fields, mass-averaged with weight rho*vx:
    Yp       = (p01 - p02) / (p02 - p2)        classic total pressure loss coefficient
    1-p02/p01                                  the dataset's own loss (pp_p0_loss in cases.csv)
p01 is the case's inlet total pressure (pp_p0_inlet, not predicted by the model) and is used for both
actual and predicted. p0 is derived from ro, vx, vr, vt, p with the case's Gamma. The product plane is
the case's pp_I_0based station unless --i is given.

    python plot_loss_coefficient.py --ckpt best.pt --split val
    python plot_loss_coefficient.py --ckpt best.pt --out loss.png --csv loss.csv
The model must have been trained on ro vx vr vt p (the default). Predicts whole meshes, so it takes a while.
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
from geotransolver_torch import GeoTransolver  # noqa: E402
from tensor_dataset import N_AXIAL_USABLE, CascadeDataset  # noqa: E402
from predict_cache import cached_predict_field  # noqa: E402
from surface_model import SurfaceModel, is_surface  # noqa: E402
from train_geotransolver import blade_surface, build_model  # noqa: E402

NEEDED = ["ro", "vx", "vr", "vt", "p"]


def p0_from(f, gamma):
    """Stagnation pressure from primitive fields (same formula as tensor_dataset.derive)."""
    v2 = f["vx"] ** 2 + f["vr"] ** 2 + f["vt"] ** 2
    m2 = v2 / (gamma * f["p"] / f["ro"])
    return f["p"] * (1 + 0.5 * (gamma - 1) * m2) ** (gamma / (gamma - 1))


def mass_avg(q, f, rt, r):
    """Mass-flux (rho*vx) weighted average over the plane. Arrays are (J, K); trapezoid in rt, then in r."""
    drt = np.abs(np.diff(rt, axis=1))
    dr = np.abs(np.diff(r.mean(axis=1)))

    def integral(y):
        across_pitch = (0.5 * (y[:, 1:] + y[:, :-1]) * drt).sum(axis=1)
        return (0.5 * (across_pitch[1:] + across_pitch[:-1]) * dr).sum()

    w = f["ro"] * f["vx"]
    return integral(w * q) / integral(w)


def losses(f, gamma, p01, rt, r):
    p02 = mass_avg(p0_from(f, gamma), f, rt, r)
    p2 = mass_avg(f["p"], f, rt, r)
    return (p01 - p02) / (p02 - p2), 1 - p02 / p01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=str(ROOT / "grid125_R23"))
    ap.add_argument("--split", choices=["all", "train", "val"], default="all", help="which cases to evaluate")
    ap.add_argument("--i", type=int, default=None, help="axial station (default: each case's pp_I_0based)")
    ap.add_argument("--cache", default=None, help="directory to cache whole-mesh predictions (shared across plots)")
    ap.add_argument("--out", default=None, help="save the figure to this file instead of showing it")
    ap.add_argument("--csv", default=None, help="also write per-case values to this CSV")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args, nm = ckpt["args"], {k: v.to(device) for k, v in ckpt["norm"].items()}
    variables = args["variables"]
    surface = is_surface(ckpt)  # blade-surface + plane model: loss is evaluated on its own plane, not the product plane
    if surface:
        sm = SurfaceModel(ckpt)
        if a.i is not None and a.i != sm.plane_i:
            sys.exit(f"this checkpoint only predicts the plane I={sm.plane_i}, not I={a.i}")
    elif any(v not in variables for v in NEEDED):
        sys.exit(f"needs variables {NEEDED}, but checkpoint has {variables}")
    else:
        model = build_model(args, len(variables)).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

    ds = CascadeDataset(a.data, variables=variables)
    n_used = len(ckpt["train_idx"]) + len(ckpt["val_idx"])
    picks = torch.linspace(0, len(ds) - 1, n_used).round().long().tolist()  # as in train_geotransolver
    split_of = {picks[i]: "train" for i in ckpt["train_idx"]} | {picks[i]: "val" for i in ckpt["val_idx"]}
    chosen = [i for i in picks if a.split == "all" or split_of[i] == a.split]
    ch = {v: variables.index(v) for v in variables}
    off_product = surface and sm.plane_i != N_AXIAL_USABLE - 1  # the dataset's pp_p0_loss is defined at I=84
    plane_name = f"plane I={sm.plane_i}" if off_product else "the product plane"

    def norm_coords(c):
        return 2 * (c - nm["c_min"]) / (nm["c_max"] - nm["c_min"]).clamp_min(1e-8) - 1

    recs = []
    for n, i in enumerate(chosen, 1):
        row, s = ds.rows[i], ds[i]
        station = sm.plane_i if surface else (a.i if a.i is not None else int(float(row["pp_I_0based"])))
        gamma, p01 = float(row["Gamma"]), float(row["pp_p0_inlet"])
        coords = s["coords"]
        if surface:
            act_all, pred_all = sm.predict_case(s, row)
            act_f = {v: act_all[v]["plane"] for v in NEEDED}
            pred_f = {v: pred_all[v]["plane"] for v in NEEDED}
        else:
            g = (2 * (s["params"].to(device) - nm["p_min"]) / (nm["p_max"] - nm["p_min"]).clamp_min(1e-8) - 1)[None, None]
            geo = norm_coords(blade_surface(coords).to(device))[None]
            full = cached_predict_field(a.cache, a.ckpt, row["case"], model,
                                        norm_coords(coords.flatten(1).T.to(device)), g, geo, args["n_points"])
            full = (full * nm["f_std"].cpu() + nm["f_mean"].cpu()).reshape(*coords.shape[1:], -1)[station].numpy()
            pred_f = {v: full[..., ch[v]] for v in NEEDED}
            act_f = {v: s["fields"][ch[v], station].numpy() for v in NEEDED}
        rt, r = coords[2, station].numpy(), coords[1, station].numpy()
        yp_a, l_a = losses(act_f, gamma, p01, rt, r)
        yp_p, l_p = losses(pred_f, gamma, p01, rt, r)
        recs.append(dict(case=row["case"], split=split_of[i], station=station, yp_actual=yp_a, yp_pred=yp_p,
                         loss_actual=l_a, loss_pred=l_p, file_loss=float(row["pp_p0_loss"])))
        print(f"\r{n}/{len(chosen)} cases", end="", flush=True)
    print()

    arr = lambda k: np.array([x[k] for x in recs])  # noqa: E731
    split = np.array([x["split"] for x in recs])
    if off_product:  # pp_p0_loss is defined at the product plane (I=84), so it is not comparable at this plane
        print(f"note: evaluated at I={sm.plane_i}, so the dataset's pp_p0_loss (product plane) does not apply")
    else:
        print(f"dataset pp_p0_loss vs recomputed from fields: max |diff| {np.abs(arr('file_loss') - arr('loss_actual')).max():.2e}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0]))
            w.writeheader()
            w.writerows(recs)
        print("wrote", a.csv)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    for ax, (name, ka, kp) in zip(axs, [("Yp = (p01 - p02) / (p02 - p2)", "yp_actual", "yp_pred"),
                                        ("1 - p02 / p01" + ("" if off_product else "  (dataset pp_p0_loss)"), "loss_actual", "loss_pred")]):
        act, pred = arr(ka), arr(kp)
        lo, hi = min(act.min(), pred.min()), max(act.max(), pred.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="perfect")
        for sp, colour in (("train", "0.6"), ("val", "tab:red")):
            m = split == sp
            if m.any():
                ax.scatter(act[m], pred[m], s=22, c=colour, alpha=0.8, label=f"{sp} ({m.sum()})")
        err = pred - act
        ss = ((act - act.mean()) ** 2).sum()
        stats = f"RMSE {np.sqrt((err**2).mean()):.2e}\nMAE {np.abs(err).mean():.2e}\nbias {err.mean():+.2e}\n" \
                f"R$^2$ {1 - (err**2).sum() / ss:.3f}"
        ax.text(0.97, 0.04, stats, transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        ax.set_xlabel("actual (CFD)")
        ax.set_ylabel("predicted")
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.legend(loc="upper left")
        print(f"{name}: RMSE {np.sqrt((err**2).mean()):.3e}, MAE {np.abs(err).mean():.3e}, bias {err.mean():+.3e}, "
              f"mean actual {act.mean():.4f}, R2 {1 - (err**2).sum() / ss:.3f}")
    fig.suptitle(f"Total pressure loss at {plane_name}, predicted vs actual ({a.split} cases, {len(recs)} scored)")
    if a.out:
        fig.savefig(a.out, dpi=150)
        print("saved", a.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
