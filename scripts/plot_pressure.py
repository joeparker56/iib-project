"""Plot CFD vs GeoTransolver static pressure on the blade surfaces and the exit plane (I=84).

Pressure is not a model output: it is derived from the predicted conserved variables with
p = (gamma - 1) * (roe - 0.5 * ro * |v|^2), exactly as it is for the CFD fields.

One figure per case, 3 rows x 3 columns:
    rows:    pressure surface, suction surface (x across, r up), exit plane (rt across, r up)
    columns: CFD, model, error (model - CFD)
CFD and model share a colour scale within each region, so they can be compared directly.

    python scripts/plot_pressure.py                         # latest run, first validation case
    python scripts/plot_pressure.py --run geotransolver_surface_01 --case g222 g343
    python scripts/plot_pressure.py --split val             # every validation case
    python scripts/plot_pressure.py --case g222 --show      # also open a window
Figures are saved in the run's own folder: results/<run>/figures/pressure/<case>.png
Get a run from Modal first (PowerShell):
    modal volume get geotransolver-runs /geotransolver_surface_01 results/
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402  (also puts the repo root on sys.path)
from run_utils import add_run_arg, load_run, resolve_run  # noqa: E402


def pressure(y, r, gamma):
    """Static pressure (Pa) from (N, 5) conserved variables in physical units; r is (N,)."""
    return T.primitives({v: y[:, i] for i, v in enumerate(T.CONSERVED)}, r, gamma)["p0"]


def regions(values, n_span, n_pitch):
    """Split a blade + exit-plane vector (ordered as in T.blade_and_exit) into its three 2D grids."""
    n_axial = T.TE - T.LE + 1
    n_face = n_axial * n_span
    return {
        "PS": values[:n_face].reshape(n_axial, n_span),
        "SS": values[n_face:2 * n_face].reshape(n_axial, n_span),
        "exit": values[2 * n_face:].reshape(n_span, n_pitch),
    }


def rel_l2(pred, true):
    """Relative L2 error of the variation about the mean (the mean pressure dominates otherwise)."""
    err = pred - true
    return np.linalg.norm(err - err.mean()) / np.linalg.norm(true - true.mean())


def plot_case(s, split_name, model, ckpt, device):
    cfg, stats = ckpt["cfg"], ckpt["stats"]

    # Encode and normalise exactly as in training, predict, and convert back to physical units
    c = T.normalise(T.encode_case(s, cfg), stats)
    y_hat, _ = T.predict(model, c, cfg, device)
    y_mean, y_std = stats["y"]
    pred_y = (y_hat.cpu() * y_std + y_mean).double().numpy()
    true_y = T.blade_and_exit(s["fields"]).double().numpy()
    r = T.blade_and_exit(s["coords"])[:, 1].double().numpy()
    gamma = s["gamma"]

    coords = s["coords"].numpy()  # (3, I, J, K): x, r, rt
    n_span, n_pitch = coords.shape[2], coords.shape[3]
    true = regions(pressure(true_y, r, gamma) / 1e3, n_span, n_pitch)  # kPa
    pred = regions(pressure(pred_y, r, gamma) / 1e3, n_span, n_pitch)
    err = {k: pred[k] - true[k] for k in true}

    blade = slice(T.LE, T.TE + 1)
    axes_xy = {  # plotting coordinates of each region, matching the (rows, cols) of the grids above
        "PS": (coords[0, blade, :, 0], coords[1, blade, :, 0]),
        "SS": (coords[0, blade, :, -1], coords[1, blade, :, -1]),
        "exit": (coords[2, T.EXIT_I], coords[1, T.EXIT_I]),
    }

    # Shared CFD/model scale per region group (blade vs exit, whose pressure ranges differ a lot),
    # and a symmetric error scale centred on zero
    groups = {"blade": ("PS", "SS"), "exit": ("exit",)}
    norms, err_norms = {}, {}
    for g, keys in groups.items():
        both = np.concatenate([d[k].ravel() for k in keys for d in (true, pred)])
        emax = max(np.abs(err[k]).max() for k in keys)
        for k in keys:
            norms[k] = Normalize(both.min(), both.max())
            err_norms[k] = TwoSlopeNorm(0, -emax, emax)

    fig, axs = plt.subplots(3, 3, figsize=(17, 13), constrained_layout=True)
    names = {"PS": "pressure surface", "SS": "suction surface", "exit": f"exit plane I={T.EXIT_I}"}
    for row, k in enumerate(("PS", "SS", "exit")):
        h, v = axes_xy[k]
        panels = (("CFD", true[k], norms[k], "viridis"),
                  ("Model", pred[k], norms[k], "viridis"),
                  ("Error (model - CFD)", err[k], err_norms[k], "RdBu_r"))
        for col, (title, z, norm, cmap) in enumerate(panels):
            ax = axs[row, col]
            ax.pcolormesh(h, v, z, norm=norm, cmap=cmap, shading="gouraud")
            ax.set_title(f"{title}: {names[k]}")
            ax.set_xlabel("rt (pitchwise)" if k == "exit" else "x (axial)")
            ax.set_ylabel("r (hub to midspan)")
            if k == "exit":
                ax.set_aspect("equal")
        axs[row, 1].text(0.02, 0.04, f"rel L2 {rel_l2(pred[k], true[k]):.2%}", transform=axs[row, 1].transAxes,
                         fontsize=10, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

    fig.colorbar(cm.ScalarMappable(norms["PS"], "viridis"), ax=axs[:2, :2], shrink=0.8, label="p (kPa)")
    fig.colorbar(cm.ScalarMappable(err_norms["PS"], "RdBu_r"), ax=axs[:2, 2], shrink=0.8, label="error (kPa)")
    fig.colorbar(cm.ScalarMappable(norms["exit"], "viridis"), ax=axs[2, :2], shrink=0.8, label="p (kPa)")
    fig.colorbar(cm.ScalarMappable(err_norms["exit"], "RdBu_r"), ax=axs[2, 2], shrink=0.8, label="error (kPa)")

    case, row = s["row"]["case"], s["row"]
    sweep, lean, theta = (float(row[k]) for k in ("sweep_w_deg", "lean_w_deg", "theta_unit"))
    fig.suptitle(f"Static pressure, case {case} ({split_name} case): "
                 f"sweep {sweep:.3g} deg, lean {lean:.3g} deg, theta_unit {theta:.3g}\n"
                 f"run {ckpt['run']}: {T.describe(cfg)}", fontsize=13)

    blade_true = np.concatenate([true["PS"].ravel(), true["SS"].ravel()])
    blade_pred = np.concatenate([pred["PS"].ravel(), pred["SS"].ravel()])
    print(f"{case}: rel L2 (variation about mean)  blade {rel_l2(blade_pred, blade_true):.3%}  "
          f"exit {rel_l2(pred['exit'], true['exit']):.3%}", flush=True)
    return fig


def main():
    ap = argparse.ArgumentParser()
    add_run_arg(ap)
    ap.add_argument("--data", default=None, help="dataset folder (default: the run's dataset)")
    ap.add_argument("--case", nargs="+", default=[], help="case ids, e.g. g222 (default: first validation case)")
    ap.add_argument("--split", choices=["val", "train"], default=None, help="also plot every case in this split")
    ap.add_argument("--show", action="store_true", help="also open each figure in a window")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = resolve_run(a.run)
    model, ckpt = load_run(run, device)
    print("run", run, flush=True)

    # Rebuild the train/val split from the checkpoint's config, so each case can be labelled
    data = a.data or T.data_dir(ckpt["cfg"])
    rows = T.read_cases(data)
    ids = [row["case"] for row in rows]
    train_idx, val_idx = T.split(len(rows), ckpt["cfg"])
    in_split = {"train": [ids[i] for i in train_idx], "val": [ids[i] for i in val_idx]}
    cases = list(a.case)
    if a.split:
        cases += [c for c in sorted(in_split[a.split]) if c not in cases]
    if not cases:
        cases = [in_split["val"][0]]
    unknown = [c for c in cases if c not in ids]
    if unknown:
        sys.exit(f"unknown case ids: {unknown}")

    out_dir = run / "figures" / "pressure"
    out_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        split_name = "validation" if case in in_split["val"] else "training"
        fig = plot_case(T.load_case(rows[ids.index(case)], data, ckpt["cfg"]["params"]), split_name, model, ckpt, device)
        fig.savefig(out_dir / f"{case}.png", dpi=150)
        print("saved", out_dir / f"{case}.png")
        if a.show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
