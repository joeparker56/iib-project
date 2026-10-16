"""Total pressure loss against blade lean at one sweep and one theta: actual vs the models.

Picks the cases of a fixed blade sweep (default 0 deg) and a fixed theta index (default 2, the middle of the five
theta values, theta_unit 0.026), sorted by lean, and plots the loss for
    actual     recomputed from the CFD fields at the product plane (I=84),
    dataset    pp_p0_loss from cases.csv (a check on the recomputation),
    each model predicted from its own fields at its plane (a surface model predicts only its own plane_i).
Two definitions, mass-averaged with weight rho*vx: 1 - p02/p01 (the dataset's) and Yp = (p01 - p02)/(p02 - p2).
Filled markers are cases in that model's validation set, hollow ones are cases it was trained on.

    python plot_loss_vs_lean.py --surface_ckpt best_surface84.pt --volume_ckpt best_global.pt
    python plot_loss_vs_lean.py --surface_ckpt best_surface84.pt --volume_ckpt best_global.pt best.pt \\
        --labels "surface 84" "volume (global params)" "volume (params both)" --cache val_plots/.cache --out lean.png
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
from tensor_dataset import N_AXIAL_USABLE, VARIABLES, CascadeDataset  # noqa: E402

PRODUCT_PLANE = N_AXIAL_USABLE - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface_ckpt", nargs="*", default=[], help="checkpoints from train_geotransolver_surface.py")
    ap.add_argument("--volume_ckpt", nargs="*", default=[], help="checkpoints from train_geotransolver.py")
    ap.add_argument("--labels", nargs="+", default=None, help="one label per checkpoint, surface ones first")
    ap.add_argument("--data", default=str(ROOT / "grid125_R23"))
    ap.add_argument("--sweep", type=float, default=0.0, help="blade sweep in degrees (a value in cases.csv)")
    ap.add_argument("--theta_index", type=int, default=2, choices=range(5),
                    help="which theta value (0-4 into the sorted values; 2 is the middle one)")
    ap.add_argument("--cache", default=None, help="directory to cache volume models' whole-mesh predictions")
    ap.add_argument("--csv", default=None, help="also write the values to this CSV")
    ap.add_argument("--out", default=None, help="save the figure to this file instead of showing it")
    a = ap.parse_args()
    if not a.surface_ckpt and not a.volume_ckpt:
        sys.exit("give at least one --surface_ckpt or --volume_ckpt")

    ds_act = CascadeDataset(a.data, variables=VARIABLES)
    thetas = sorted({float(r["theta_unit"]) for r in ds_act.rows})
    picks = sorted([i for i, r in enumerate(ds_act.rows)
                    if float(r["sweep_w_deg"]) == a.sweep and int(r["k_theta"]) == a.theta_index],
                   key=lambda i: float(ds_act.rows[i]["lean_w_deg"]))
    if not picks:
        sys.exit(f"no cases with sweep {a.sweep} at theta index {a.theta_index}")
    n = len(ds_act)
    leans = [float(ds_act.rows[i]["lean_w_deg"]) for i in picks]
    print(f"sweep {a.sweep:g} deg, theta {thetas[a.theta_index]:.3f} (index {a.theta_index}): "
          f"cases {[ds_act.rows[i]['case'] for i in picks]}, lean {leans}")

    # ---- models: (label, kind, checkpoint, predictor) ----
    paths = a.surface_ckpt + a.volume_ckpt
    labels = a.labels or [f"surface ({Path(p).stem})" for p in a.surface_ckpt] + \
        [f"volume ({Path(p).stem})" for p in a.volume_ckpt]
    if len(labels) != len(paths):
        sys.exit("--labels needs one label per checkpoint (surface ones first)")
    models = []
    for path, label in zip(paths, labels):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if path in a.surface_ckpt:
            if not is_surface(ck):
                sys.exit(f"{path} is not a surface-model checkpoint")
            sm = SurfaceModel(ck)
            models.append(dict(label=label, ck=ck, surface=sm, plane=sm.plane_i,
                               ds=CascadeDataset(a.data, variables=sm.variables)))
        else:
            if is_surface(ck):
                sys.exit(f"{path} is a surface-model checkpoint; pass it with --surface_ckpt")
            vm = VolumeModel(path, ck, a.cache)
            models.append(dict(label=label, ck=ck, volume=vm, plane=PRODUCT_PLANE,
                               ds=CascadeDataset(a.data, variables=vm.variables)))
        if models[-1]["plane"] != PRODUCT_PLANE:
            print(f"note: {label} predicts plane I={models[-1]['plane']}, not the product plane "
                  f"I={PRODUCT_PLANE}; its loss is at that plane and is not directly comparable")

    # ---- loss per case ----
    res = {"actual": [], "dataset": []}
    for m in models:
        res[m["label"]] = []
    rows_out = []
    for k, i in enumerate(picks, 1):
        row, case = ds_act.rows[i], ds_act.rows[i]["case"]
        gamma, p01 = float(row["Gamma"]), float(row["pp_p0_inlet"])
        s = ds_act[i]
        coords = s["coords"]

        def plane_fields(plane):
            return ({v: s["fields"][VARIABLES.index(v), plane].numpy() for v in NEEDED},
                    coords[2, plane].numpy(), coords[1, plane].numpy())

        f, rt, r = plane_fields(PRODUCT_PLANE)
        res["actual"].append(losses(f, gamma, p01, rt, r))
        res["dataset"].append((np.nan, float(row["pp_p0_loss"])))
        rec = dict(case=case, lean=float(row["lean_w_deg"]), actual_yp=res["actual"][-1][0],
                   actual_loss=res["actual"][-1][1], dataset_loss=res["dataset"][-1][1])
        for m in models:
            plane = m["plane"]
            if "surface" in m:
                _, pred_all = m["surface"].predict_case(m["ds"][i], m["ds"].rows[i])
                pf = {v: pred_all[v]["plane"] for v in NEEDED}
            else:
                pf = {v: arr[plane] for v, arr in m["volume"].fields(m["ds"][i], case, gamma).items() if v in NEEDED}
            _, rt_m, r_m = plane_fields(plane)
            res[m["label"]].append(losses(pf, gamma, p01, rt_m, r_m))
            rec[f"{m['label']} yp"], rec[f"{m['label']} loss"] = res[m["label"]][-1]
            rec[f"{m['label']} split"] = split_label(m["ck"], n, i)
        rows_out.append(rec)
        print(f"\r{k}/{len(picks)} cases", end="", flush=True)
    print()

    print(f"\n{'case':6s} {'lean':>6s} {'actual':>9s} {'dataset':>9s} " + " ".join(f"{m['label'][:22]:>22s}" for m in models))
    for rec in rows_out:
        print(f"{rec['case']:6s} {rec['lean']:6.0f} {rec['actual_loss']:9.5f} {rec['dataset_loss']:9.5f} "
              + " ".join(f"{rec[m['label'] + ' loss']:>13.5f} ({rec[m['label'] + ' split']:5s})" for m in models))
    act = np.array([x[1] for x in res["actual"]])
    for m in models:
        p = np.array([x[1] for x in res[m["label"]]])
        print(f"{m['label']}: mean error {np.mean(p - act):+.2e}, RMSE {np.sqrt(np.mean((p - act) ** 2)):.2e}, "
              f"correlation with actual over lean {np.corrcoef(p, act)[0, 1]:.2f}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
        print("wrote", a.csv)

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    for ax, (title, idx) in zip(axs, (("1 - p02 / p01 (dataset's loss)", 1), ("Yp = (p01 - p02) / (p02 - p2)", 0))):
        ax.plot(leans, [x[idx] for x in res["actual"]], "k-o", lw=2.2, ms=7, label=f"actual (CFD, I={PRODUCT_PLANE})", zorder=5)
        if idx == 1:
            ax.plot(leans, [x[1] for x in res["dataset"]], ":", color="0.45", lw=1.6, marker="x",
                    label="dataset pp_p0_loss", zorder=4)
        for j, m in enumerate(models):
            c = colours[j % len(colours)]
            tag = "" if m["plane"] == PRODUCT_PLANE else f" (I={m['plane']})"
            ax.plot(leans, [x[idx] for x in res[m["label"]]], "-", color=c, lw=1.6, label=m["label"] + tag)
            for lean, x, rec in zip(leans, res[m["label"]], rows_out):
                val = rec[m["label"] + " split"] == "val"
                ax.plot(lean, x[idx], "o", ms=8, color=c, mfc=c if val else "white", mew=1.8)
        ax.set_xlabel("blade lean (deg)")
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.set_xticks(leans)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(f"Loss vs blade lean, sweep {a.sweep:g} deg, theta {thetas[a.theta_index]:.3f} "
                 f"(index {a.theta_index}); filled = validation case for that model, hollow = trained on")
    if a.out:
        fig.savefig(a.out, dpi=150)
        print("saved", a.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
