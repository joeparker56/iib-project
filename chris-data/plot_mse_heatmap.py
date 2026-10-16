"""Heat map of model MSE over the design space: sweep on x, lean on y (theta averaged, or one panel per theta).

    python plot_mse_heatmap.py --ckpt best.pt
    python plot_mse_heatmap.py --ckpt best.pt --per_theta --channel p --split val --out mse.png
MSE is in normalised (per-channel standardised) units, as in training, on a fixed random subset of mesh points.
Get the checkpoint from Modal first:  modal volume get geotransolver-runs /geotransolver/best.pt .
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from geotransolver_torch import GeoTransolver  # noqa: E402
from surface_model import SurfaceModel, is_surface  # noqa: E402
from tensor_dataset import CascadeDataset  # noqa: E402
from train_geotransolver import blade_surface, build_model, model_inputs  # noqa: E402

CHUNK = 16384  # points per forward pass (matches training --n_points)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=str(ROOT / "grid125_R23"))
    ap.add_argument("--split", choices=["all", "train", "val"], default="all", help="which cases to score")
    ap.add_argument("--channel", default=None, help="single variable to score (default: mean over all channels)")
    ap.add_argument("--n_points", type=int, default=65536, help="fixed random mesh points per case")
    ap.add_argument("--per_theta", action="store_true", help="one panel per theta instead of averaging over theta")
    ap.add_argument("--theta_index", type=int, default=None, choices=range(5),
                    help="plot a single theta slice (0-4 into the sorted theta values; 2 is the middle one)")
    ap.add_argument("--region", choices=["all", "blade", "plane"], default="all",
                    help="blade-surface + plane models only: score all their points, just the blade or just the plane")
    ap.add_argument("--out", default=None, help="save to this file instead of showing")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args, nm = ckpt["args"], {k: v.to(device) for k, v in ckpt["norm"].items()}
    variables = args["variables"]
    if a.channel is not None and a.channel not in variables:
        sys.exit(f"channel {a.channel!r} not in checkpoint variables {variables}")

    surface = is_surface(ckpt)  # blade-surface + plane model: scored on exactly the points it predicts
    if surface:
        sm = SurfaceModel(ckpt)
    else:
        model = build_model(args, len(variables)).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

    ds = CascadeDataset(a.data, variables=variables)
    # train_geotransolver trains on `picks` (evenly spread cases); train_idx/val_idx index into that list
    n_used = len(ckpt["train_idx"]) + len(ckpt["val_idx"])
    picks = torch.linspace(0, len(ds) - 1, n_used).round().long().tolist()
    chosen = {"all": picks, "train": [picks[i] for i in ckpt["train_idx"]], "val": [picks[i] for i in ckpt["val_idx"]]}[a.split]
    chans = list(range(len(variables))) if a.channel is None else [variables.index(a.channel)]

    def norm_coords(c):
        return 2 * (c - nm["c_min"]) / (nm["c_max"] - nm["c_min"]).clamp_min(1e-8) - 1

    pts = torch.randperm(1, generator=torch.Generator().manual_seed(0))  # placeholder, set per size below
    mse, idx = [], []
    with torch.no_grad():
        for n, i in enumerate(chosen, 1):
            s = ds[i]
            if surface:
                pred_p, act_p, _, nb = sm.predict_points(s, ds.rows[i])
                sl = {"all": slice(None), "blade": slice(0, nb), "plane": slice(nb, None)}[a.region]
                e = (pred_p - act_p)[sl][:, chans] / sm.norm["f_std"].numpy()[chans]  # standardised units
                mse.append(float((e**2).mean()))
                idx.append(s["index"].tolist())  # (i_sweep, j_lean, k_theta)
                print(f"\r{n}/{len(chosen)} cases", end="", flush=True)
                continue
            coords = s["coords"].flatten(1).T.to(device)  # (N, 3)
            if len(pts) <= 1:
                pts = torch.randperm(coords.shape[0], generator=torch.Generator().manual_seed(0))[: a.n_points]
            pts_d = pts.to(device)
            c = norm_coords(coords[pts_d])
            geo = norm_coords(blade_surface(s["coords"]).to(device))[None]
            g = (2 * (s["params"].to(device) - nm["p_min"]) / (nm["p_max"] - nm["p_min"]).clamp_min(1e-8) - 1)[None, None]
            pred = []
            for j in range(0, len(c), CHUNK):
                cc = c[j : j + CHUNK][None]
                local, gl = model_inputs(model, cc, g)
                pred.append(model(local, geometry=geo, global_embedding=gl)[0])
            y = (s["fields"].flatten(1).T.to(device)[pts_d] - nm["f_mean"]) / nm["f_std"]
            err = (torch.cat(pred) - y)[:, chans]
            mse.append((err**2).mean().item())
            idx.append(s["index"].tolist())  # (i_sweep, j_lean, k_theta)
            print(f"\r{n}/{len(chosen)} cases", end="", flush=True)
    print()

    mse, idx = np.array(mse), np.array(idx)
    axes_vals = {  # physical values for each axis index, read from the case table
        k: sorted({float(r[k]) for r in ds.rows}) for k in ("sweep_w_deg", "lean_w_deg", "theta_unit")
    }
    ns, nl, nt = (len(axes_vals[k]) for k in ("sweep_w_deg", "lean_w_deg", "theta_unit"))

    grid = np.full((nt, nl, ns), np.nan)  # [theta, lean (y), sweep (x)]
    for m, (i, j, k) in zip(mse, idx):
        grid[k, j, i] = m
    if a.theta_index is not None:
        k = a.theta_index
        panels = [(f"theta = {axes_vals['theta_unit'][k]:.3f}", grid[k])]
    elif a.per_theta:
        panels = [(f"theta = {axes_vals['theta_unit'][k]:.3f}", grid[k]) for k in range(nt)]
    else:
        panels = [("mean over theta", np.nanmean(grid, axis=0))]
    shared = a.per_theta and a.theta_index is None  # one colour scale across the per-theta panels

    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    ncols = len(panels)
    fig, axs = plt.subplots(1, ncols, figsize=(4.6 * ncols + 1.5, 4.6), squeeze=False, constrained_layout=True)
    for ax, (title, z) in zip(axs[0], panels):
        im = ax.imshow(z, origin="lower", cmap="magma", vmin=vmin if shared else np.nanmin(z),
                       vmax=vmax if shared else np.nanmax(z))
        ax.set_xticks(range(ns), [f"{v:g}" for v in axes_vals["sweep_w_deg"]])
        ax.set_yticks(range(nl), [f"{v:g}" for v in axes_vals["lean_w_deg"]])
        ax.set_xlabel("sweep (deg)")
        ax.set_ylabel("lean (deg)")
        ax.set_title(title)
        mid = 0.5 * (vmin + vmax) if shared else 0.5 * (np.nanmin(z) + np.nanmax(z))
        for (r, c_), v in np.ndenumerate(z):
            if np.isfinite(v):
                ax.text(c_, r, f"{v:.2g}", ha="center", va="center", fontsize=8,
                        color="white" if v < mid else "black")
        if not shared:
            fig.colorbar(im, ax=ax, label="MSE (normalised)")
    if shared:
        fig.colorbar(im, ax=axs[0].tolist(), label="MSE (normalised)", shrink=0.8)
    what = (a.channel or "all channels") + (f", {a.region} points" if surface else "")
    fig.suptitle(f"Test MSE over design space ({what}, {a.split} cases, {len(mse)} cases scored)")
    print(f"overall MSE {mse.mean():.4e}; worst case MSE {mse.max():.4e}")
    if a.out:
        fig.savefig(a.out, dpi=150)
        print("saved", a.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
