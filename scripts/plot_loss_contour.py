"""Contour plots of the exit-plane loss coefficient against sweep and lean angle: CFD vs model.

The boundary-layer thickness (theta_unit) is fixed at the middle of its 5 values by default, leaving
the 5 x 5 sweep x lean grid of cases. For each case, from the exit-plane (I=84) fields:
    Yp = (p01 - p02) / (p02 - p2)     total pressure loss coefficient
where p02 and p2 are mass-averaged (weight rho*vx) stagnation and static pressure over the plane, and
p01 is the case's inlet stagnation pressure (pp_p0_inlet in cases.csv, not predicted by the model,
so the same value is used for CFD and model). The model's Yp comes from its predicted conserved
fields, converted to p and p0 exactly as for the CFD.

Saves into the run folder:
    results/<run>/figures/loss_contour_theta<value>.png
    results/<run>/figures/loss_contour_theta<value>.csv   per-case values, including the QoI head's p0 loss

    python scripts/plot_loss_contour.py                          # latest run, middle theta_unit
    python scripts/plot_loss_contour.py --run geotransolver_surface_01 --theta 0.002
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402  (also puts the repo root on sys.path)
from run_utils import add_run_arg, load_run, resolve_run  # noqa: E402


def mass_avg(q, f, rt, r):
    """Mass-flux (rho*vx) weighted average over the plane. Arrays are (J, K); trapezoid in rt, then in r.
"""
    drt = np.abs(np.diff(rt, axis=1))
    dr = np.abs(np.diff(r.mean(axis=1)))

    def integral(y):
        across_pitch = (0.5 * (y[:, 1:] + y[:, :-1]) * drt).sum(axis=1)
        return (0.5 * (across_pitch[1:] + across_pitch[:-1]) * dr).sum()

    w = f["ro"] * f["vx"]
    return integral(w * q) / integral(w)


def exit_losses(y, coords, gamma, p01, n_blade):
    """(Yp, 1 - p02/p01) from (N, 5) blade + exit-plane conserved fields in physical units."""
    rt, r = coords[2, T.EXIT_I], coords[1, T.EXIT_I]  # (J, K)
    plane = y[n_blade:].reshape(*r.shape, len(T.CONSERVED))
    f = {v: plane[..., i] for i, v in enumerate(T.CONSERVED)}
    f.update(T.primitives(f, r, gamma))  # adds vx, vr, vt, p, p0
    p02, p2 = mass_avg(f["p0"], f, rt, r), mass_avg(f["p"], f, rt, r)
    return (p01 - p02) / (p02 - p2), 1 - p02 / p01


def main():
    ap = argparse.ArgumentParser()
    add_run_arg(ap)
    ap.add_argument("--data", default=None, help="dataset folder (default: the run's dataset)")
    ap.add_argument("--theta", type=float, default=None, help="theta_unit to plot (default: the middle value)")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = resolve_run(a.run)
    model, ckpt = load_run(run, device)
    cfg, stats = ckpt["cfg"], ckpt["stats"]
    print("run", run, flush=True)

    data = a.data or T.data_dir(cfg)
    cases_csv = T.read_cases(data)
    ids = [row["case"] for row in cases_csv]
    _, val_idx = T.split(len(cases_csv), cfg)
    val_ids = {ids[i] for i in val_idx}
    thetas = sorted({float(row["theta_unit"]) for row in cases_csv})
    theta = thetas[len(thetas) // 2] if a.theta is None else min(thetas, key=lambda t: abs(t - a.theta))
    sweeps = sorted({float(row["sweep_w_deg"]) for row in cases_csv})
    leans = sorted({float(row["lean_w_deg"]) for row in cases_csv})
    if len(thetas) * len(sweeps) * len(leans) != len(cases_csv):
        sys.exit(f"{cfg['dataset']} is not a sweep x lean x theta_unit grid, so it can't be contoured this way")

    # --- Evaluate every case at this theta ---------------------------------------------------
    y_mean, y_std = stats["y"]
    q_mean, q_std = stats["q"]
    yp_cfd, yp_model = np.full((len(leans), len(sweeps)), np.nan), np.full((len(leans), len(sweeps)), np.nan)
    rows = []
    for row in cases_csv:
        if abs(float(row["theta_unit"]) - theta) > 1e-9:
            continue
        s = T.load_case(row, data, cfg["params"])
        c = T.normalise(T.encode_case(s, cfg), stats)
        y_hat, q_hat = T.predict(model, c, cfg, device)
        pred_y = (y_hat.cpu() * y_std + y_mean).double().numpy()
        true_y = T.blade_and_exit(s["fields"]).double().numpy()
        coords = s["coords"].double().numpy()
        gamma, p01 = s["gamma"], float(row["pp_p0_inlet"])
        yp_c, loss_c = exit_losses(true_y, coords, gamma, p01, c["n_blade"])
        yp_m, loss_m = exit_losses(pred_y, coords, gamma, p01, c["n_blade"])
        sweep, lean = float(row["sweep_w_deg"]), float(row["lean_w_deg"])
        yp_cfd[leans.index(lean), sweeps.index(sweep)] = yp_c
        yp_model[leans.index(lean), sweeps.index(sweep)] = yp_m
        out = {"case": row["case"], "split": "val" if row["case"] in val_ids else "train",
               "sweep_deg": sweep, "lean_deg": lean, "theta_unit": theta,
               "Yp_cfd": yp_c, "Yp_model": yp_m,
               "p0_loss_csv": float(row["pp_p0_loss"]), "p0_loss_cfd_fields": loss_c, "p0_loss_model_fields": loss_m}
        if "pp_p0_loss" in cfg["qoi"]:  # the QoI head's direct prediction of the same quantity
            k = list(cfg["qoi"]).index("pp_p0_loss")
            out["p0_loss_model_head"] = float(q_hat[k].cpu() * q_std[k] + q_mean[k])
        rows.append(out)
        print(f"{row['case']} sweep {sweep:+5.0f} lean {lean:+5.0f}  Yp CFD {yp_c:.4f}  model {yp_m:.4f}", flush=True)

    # Sanity check: our CFD 1 - p02/p01 should reproduce the dataset's pp_p0_loss
    check = max(abs(r["p0_loss_cfd_fields"] - r["p0_loss_csv"]) for r in rows)
    print(f"max |1 - p02/p01 (CFD fields) - pp_p0_loss (cases.csv)| = {check:.2e}")

    # --- Plot ---------------------------------------------------------------------------------
    S, L = np.meshgrid(sweeps, leans)  # (lean, sweep), matching the Yp arrays
    err = yp_model - yp_cfd
    # CFD and model share levels set by the CFD range, so the CFD pattern is always readable; model
    # values outside it saturate (extend="both") and the error panel shows by how much
    levels = np.linspace(np.nanmin(yp_cfd), np.nanmax(yp_cfd), 15)
    emax = np.nanmax(np.abs(err))
    is_val = np.array([r["split"] == "val" for r in rows])
    pts = np.array([[r["sweep_deg"], r["lean_deg"]] for r in rows])

    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True)
    panels = (("CFD", yp_cfd, dict(levels=levels, cmap="viridis", extend="both")),
              ("Model (from predicted exit-plane fields)", yp_model, dict(levels=levels, cmap="viridis", extend="both")),
              ("Error (model - CFD)", err, dict(levels=np.linspace(-emax, emax, 15), cmap="RdBu_r",
                                                norm=TwoSlopeNorm(0, -emax, emax))))
    for ax, (title, z, kw) in zip(axs, panels):
        cs = ax.contourf(S, L, z, **kw)
        ax.contour(S, L, z, levels=kw["levels"], colors="k", linewidths=0.4, alpha=0.4)
        fig.colorbar(cs, ax=ax, shrink=0.9, label="Yp" if "Error" not in title else "Yp error")
        # mark where the cases are, distinguishing those the model was trained on
        ax.scatter(*pts[~is_val].T, s=14, c="k", zorder=3, clip_on=False)
        ax.scatter(*pts[is_val].T, s=70, facecolors="none", edgecolors="k", linewidths=1.5, zorder=3, clip_on=False)
        ax.set_title(title)
        ax.set_xlabel("sweep angle (deg)")
        ax.set_ylabel("lean angle (deg)")
        ax.set_xticks(sweeps)
        ax.set_yticks(leans)
        ax.set_aspect("equal")
    axs[0].legend(handles=[
        Line2D([], [], marker="o", ls="", color="k", markersize=4, label="training case"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="k", markersize=8, label="validation case"),
    ], loc="upper left", fontsize=9, framealpha=0.9)
    rmse = np.sqrt(np.nanmean(err**2))
    fig.suptitle(f"Loss coefficient Yp = (p01 - p02)/(p02 - p2) at exit plane I={T.EXIT_I}, "
                 f"theta_unit = {theta:g}   (RMSE {rmse:.4f}, max |err| {emax:.4f})\n"
                 f"run {ckpt['run']}: {T.describe(cfg)}", fontsize=13)

    out_dir = run / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    png, table = (out_dir / f"loss_contour_theta{theta:g}.{ext}" for ext in ("png", "csv"))
    fig.savefig(png, dpi=150)
    with open(table, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", png, "and", table)


if __name__ == "__main__":
    main()
