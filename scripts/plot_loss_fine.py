"""Fine sweep x lean map of the model's p0 loss, side by side with the raw CFD (grid-type datasets only).

The CFD only exists on the 15 deg grid of geometries (5 sweeps x 5 leans). The model can be evaluated at
any sweep and lean, but it needs that geometry's mesh as input (query points and wall point cloud), so
in-between meshes are interpolated from the CFD meshes, which all share the same (I, J, K) structure:
    lean:  cubic spline through the 5 lean values (reproduces a left-out mesh to ~1e-7)
    sweep: linear between the two neighbouring sweeps (a left-out mesh 30 deg from its neighbours is
           reproduced to ~1e-2 on the blade and ~2.4e-2 on the exit plane, axial chord ~1; at the real
           15 deg spacing the error is several times smaller). Linear rather than cubic because the exit
           plane's position has a kink at sweep 0: it sits 0.5 c_ax behind the furthest-back TE point.
The plotted quantity is p0 loss = 1 - p02/p01 at the product plane: the QoI head's prediction for the
model, and pp_p0_loss from cases.csv for the CFD. The head is far more accurate than the loss derived
from the predicted fields (compare plot_loss_contour.py).

Saves results/<run>/figures/loss_fine_theta<value>.png and .csv

    python scripts/plot_loss_fine.py --run surrogate_dataset
    python scripts/plot_loss_fine.py --run surrogate_dataset --step 2.5 --theta 0.002
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
from run_utils import add_run_arg, load_run, resolve_run  # noqa: E402


def load_coords(data, grid_file):
    with np.load(Path(data) / "grids" / grid_file) as g:
        return np.stack([g[k][:T.N_AXIAL] for k in ("x", "r", "rt")]).astype(np.float64)


def predict_p0_loss(model, ckpt, coords, inputs, device):
    """The QoI head's p0 loss for one geometry. coords: (3, I, J, K) mesh; inputs: dict of design inputs."""
    cfg, stats = ckpt["cfg"], ckpt["stats"]
    # encode_case expects a loaded case; the fields and QoIs are only training targets, so zeros will do
    s = {
        "row": {"case": "interp", **{k: "0" for k in cfg["qoi"]}},
        "fields": torch.zeros(len(T.CONSERVED), *coords.shape[1:]),
        "coords": torch.from_numpy(coords.astype(np.float32)),
        "params": torch.tensor([float(inputs[k]) for k in cfg["params"]]),
    }
    c = T.normalise(T.encode_case(s, cfg), stats)
    _, q_hat = T.predict(model, c, cfg, device)
    k = list(cfg["qoi"]).index("pp_p0_loss")
    q_mean, q_std = stats["q"]
    return float(q_hat[k].cpu() * q_std[k] + q_mean[k])


def main():
    ap = argparse.ArgumentParser()
    add_run_arg(ap)
    ap.add_argument("--data", default=None, help="dataset folder (default: the run's dataset)")
    ap.add_argument("--theta", type=float, default=None, help="theta_unit to plot (default: the middle value)")
    ap.add_argument("--step", type=float, default=5.0, help="sweep / lean spacing of the model map (deg)")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = resolve_run(a.run)
    model, ckpt = load_run(run, device)
    cfg = ckpt["cfg"]
    if "pp_p0_loss" not in cfg["qoi"]:
        sys.exit("this run's QoI head does not predict pp_p0_loss")
    print("run", run, flush=True)

    data = a.data or T.data_dir(cfg)
    rows = T.read_cases(data)
    thetas = sorted({float(r["theta_unit"]) for r in rows})
    sweeps = sorted({float(r["sweep_w_deg"]) for r in rows})
    leans = sorted({float(r["lean_w_deg"]) for r in rows})
    if len(thetas) * len(sweeps) * len(leans) != len(rows):
        sys.exit(f"{cfg['dataset']} is not a sweep x lean x theta_unit grid")
    theta = thetas[len(thetas) // 2] if a.theta is None else min(thetas, key=lambda t: abs(t - a.theta))

    # Inputs other than sweep / lean (e.g. theta_unit) are taken from a CFD case at this theta
    at_theta = [r for r in rows if abs(float(r["theta_unit"]) - theta) < 1e-9]
    template = at_theta[0]
    cfd = np.full((len(leans), len(sweeps)), np.nan)
    for r in at_theta:
        cfd[leans.index(float(r["lean_w_deg"])), sweeps.index(float(r["sweep_w_deg"]))] = float(r["pp_p0_loss"])

    # Every CFD mesh, indexed [sweep][lean]
    grid_file = {(float(r["sweep_w_deg"]), float(r["lean_w_deg"])): r["grid"] for r in rows}
    meshes = np.stack([np.stack([load_coords(data, grid_file[(s, l)]) for l in leans]) for s in sweeps])

    # Sanity check of the mesh interpolation's effect on the model: rebuild the (0, 0) mesh linearly from
    # its sweep neighbours 15 deg either side (a 30 deg gap, worse than any interpolation used below)
    if 0.0 in sweeps and 0.0 in leans:
        i, j = sweeps.index(0.0), leans.index(0.0)
        inputs = {**template, "sweep_w_deg": 0.0, "lean_w_deg": 0.0}
        exact = predict_p0_loss(model, ckpt, meshes[i, j], inputs, device)
        rebuilt = predict_p0_loss(model, ckpt, 0.5 * (meshes[i - 1, j] + meshes[i + 1, j]), inputs, device)
        print(f"check: model p0 loss at sweep 0, lean 0 with the real mesh {exact:.5f}, with a mesh interpolated "
              f"across 30 deg {rebuilt:.5f} (difference {rebuilt - exact:+.1e})", flush=True)

    # --- Model on the fine grid ---------------------------------------------------------------
    fine_s = np.arange(sweeps[0], sweeps[-1] + 1e-9, a.step)
    fine_l = np.arange(leans[0], leans[-1] + 1e-9, a.step)
    lean_spline = CubicSpline(leans, meshes, axis=1)  # cubic in lean, for every CFD sweep at once
    model_map = np.full((len(fine_l), len(fine_s)), np.nan)
    out = []
    for jl, lean in enumerate(fine_l):
        at_lean = lean_spline(lean)  # (n_sweeps, 3, I, J, K)
        for js, sweep in enumerate(fine_s):
            hi = min(np.searchsorted(sweeps, sweep), len(sweeps) - 1)  # linear between neighbouring sweeps
            lo = max(hi - 1, 0)
            t = 0.0 if hi == lo else (sweep - sweeps[lo]) / (sweeps[hi] - sweeps[lo])
            coords = (1 - t) * at_lean[lo] + t * at_lean[hi]
            loss = predict_p0_loss(model, ckpt, coords, {**template, "sweep_w_deg": sweep, "lean_w_deg": lean}, device)
            model_map[jl, js] = loss
            on_cfd = sweep in sweeps and lean in leans
            out.append({"sweep_deg": sweep, "lean_deg": lean, "theta_unit": theta, "p0_loss_model": loss,
                        "p0_loss_cfd": cfd[leans.index(lean), sweeps.index(sweep)] if on_cfd else ""})
        print(f"lean {lean:+5.1f}: done {len(fine_s)} sweeps", flush=True)

    # Error at the CFD geometries, where the model used the real meshes
    nodes = [(r["p0_loss_model"], r["p0_loss_cfd"]) for r in out if r["p0_loss_cfd"] != ""]
    rmse = np.sqrt(np.mean([(m - c) ** 2 for m, c in nodes]))
    print(f"RMSE at the {len(nodes)} CFD geometries: {rmse:.2e} (CFD spread {np.nanmax(cfd) - np.nanmin(cfd):.2e})")

    # --- Plot: CFD on its 15 deg grid | model on the fine grid, same colour levels -------------------
    levels = MaxNLocator(nbins=16).tick_values(min(np.nanmin(cfd), model_map.min()) * 100,
                                               max(np.nanmax(cfd), model_map.max()) * 100) / 100  # round values
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    panels = (("CFD (15 deg grid of cases)", sweeps, leans, cfd),
              (f"Model QoI head ({a.step:g} deg grid)", fine_s, fine_l, model_map))
    for ax, (title, xs, ys, z) in zip(axs, panels):
        S, L = np.meshgrid(xs, ys)
        cs = ax.contourf(S, L, z * 100, levels=levels * 100, cmap="viridis")
        ax.contour(S, L, z * 100, levels=levels * 100, colors="k", linewidths=0.4, alpha=0.4)
        ax.scatter(*np.meshgrid(sweeps, leans), s=14, c="k", zorder=3, clip_on=False)  # CFD geometries
        ax.set_title(title)
        ax.set_xlabel("sweep angle (deg)")
        ax.set_ylabel("lean angle (deg)")
        ax.set_xticks(sweeps)
        ax.set_yticks(leans)
        ax.set_aspect("equal")
    fig.colorbar(cs, ax=axs, shrink=0.9, label="p0 loss = 1 - p02/p01 (%)")
    fig.suptitle(f"Product-plane p0 loss, theta_unit = {theta:g}   (model RMSE at CFD geometries {rmse * 100:.3f} %)\n"
                 f"run {ckpt['run']}: {T.describe(cfg)}", fontsize=12)

    out_dir = run / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    png, table = (out_dir / f"loss_fine_theta{theta:g}.{ext}" for ext in ("png", "csv"))
    fig.savefig(png, dpi=150)
    with open(table, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("saved", png, "and", table)


if __name__ == "__main__":
    main()
