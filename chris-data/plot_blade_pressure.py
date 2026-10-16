"""Plot actual vs predicted vs error blade pressure on the 3D blade surface (PS + SS, LE..TE, all spans).

    python plot_blade_pressure.py --ckpt best.pt                 # first validation case
    python plot_blade_pressure.py --ckpt best.pt --case g222 --out p.png
Batches (the model and dataset are loaded once; one PNG per case in --out_dir):
    python plot_blade_pressure.py --ckpt best.pt --split val --out_dir blade_val
    python plot_blade_pressure.py --ckpt best.pt --case g001 g134 g222 --out_dir blade_some --skip_existing
Get the checkpoint from Modal first:  modal volume get geotransolver-runs /geotransolver/best.pt .
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
from tensor_dataset import CascadeDataset  # noqa: E402
from predict_cache import cached_predict_field  # noqa: E402
from surface_model import SurfaceModel, is_surface  # noqa: E402
from train_geotransolver import LE_I, TE_I, blade_surface, build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=str(ROOT / "grid125_R23"))
    ap.add_argument("--case", nargs="+", default=None, help="one or more case ids (default: first validation case)")
    ap.add_argument("--split", choices=["val", "train", "all"], default=None,
                    help="plot every case in this split (added to any --case ids); needs --out_dir")
    ap.add_argument("--plot", choices=["2d", "3d"], default="2d", help="2d flat maps (default) or 3d surface plots")
    ap.add_argument("--cache", default=None, help="directory to cache whole-mesh predictions (shared across plots)")
    ap.add_argument("--out", default=None, help="save a single case to this file instead of showing")
    ap.add_argument("--out_dir", default=None, help="save <case>.png for each case here (needed for several cases)")
    ap.add_argument("--skip_existing", action="store_true", help="with --out_dir, skip cases whose PNG already exists")
    a = ap.parse_args()

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args, nm = ckpt["args"], ckpt["norm"]
    surface = is_surface(ckpt)  # blade-surface + plane model: pressure is derived from the conserved fields
    if not surface and "p" not in args["variables"]:
        sys.exit(f"checkpoint was not trained on pressure (variables: {args['variables']})")
    ch = None if surface else args["variables"].index("p")

    ds = CascadeDataset(a.data, variables=args["variables"])
    ids = [r["case"] for r in ds.rows]
    # train_geotransolver picks cases with linspace over the dataset; train_idx/val_idx index that subset
    picks = torch.linspace(0, len(ds) - 1, len(ckpt["train_idx"]) + len(ckpt["val_idx"])).round().long().tolist()
    in_split = {"val": [ids[picks[i]] for i in ckpt["val_idx"]], "train": [ids[picks[i]] for i in ckpt["train_idx"]]}
    in_split["all"] = in_split["val"] + in_split["train"]
    cases = list(a.case or [])
    if a.split:
        cases += [c for c in sorted(in_split[a.split]) if c not in cases]
    if not cases:
        cases = [in_split["val"][0]]
    unknown = [c for c in cases if c not in ids]
    if unknown:
        sys.exit(f"unknown case ids: {unknown}")
    if len(cases) > 1 and not a.out_dir:
        sys.exit("several cases need --out_dir (one PNG per case)")
    if a.out and len(cases) > 1:
        sys.exit("--out is for a single case; use --out_dir")

    if surface:
        sm = SurfaceModel(ckpt)
    else:
        model = build_model(args, len(args["variables"]))
        model.load_state_dict(ckpt["model"])
        model.eval()

    def norm_coords(c):
        return 2 * (c - nm["c_min"]) / (nm["c_max"] - nm["c_min"]).clamp_min(1e-8) - 1

    def plot_case(case):
        s = ds[ids.index(case)]
        print("case", case, "params (sweep, lean, theta):", s["params"].tolist(), flush=True)
        coords = s["coords"]  # (3, I, J, K)
        faces = {"PS": coords[:, LE_I : TE_I + 1, :, 0], "SS": coords[:, LE_I : TE_I + 1, :, -1]}  # (3, nI, J)
        if surface:  # the model predicts exactly these blade points; pressure comes from the conserved fields
            act_all, pred_all = sm.predict_case(s, ds.rows[ids.index(case)])
            actual = {k: act_all["p"][k] for k in ("PS", "SS")}
            pred = {k: pred_all["p"][k] for k in ("PS", "SS")}
        else:
            actual = {"PS": s["fields"][ch, LE_I : TE_I + 1, :, 0], "SS": s["fields"][ch, LE_I : TE_I + 1, :, -1]}
            g = (2 * (s["params"] - nm["p_min"]) / (nm["p_max"] - nm["p_min"]).clamp_min(1e-8) - 1)[None, None]
            geo = norm_coords(blade_surface(coords))[None]
            # predict the whole mesh in training-like random chunks, then read off the blade faces
            full = cached_predict_field(a.cache, a.ckpt, case, model, norm_coords(coords.flatten(1).T), g, geo,
                                        args["n_points"])[:, ch]
            full = (full * nm["f_std"][ch] + nm["f_mean"][ch]).reshape(coords.shape[1:]).numpy()  # (I, J, K)
            pred = {"PS": full[LE_I : TE_I + 1, :, 0], "SS": full[LE_I : TE_I + 1, :, -1]}
            actual = {k: v.numpy() for k, v in actual.items()}
        err = {k: pred[k] - actual[k] for k in pred}

        # The grid is one passage: K=0 is the PS of one blade, K=60 the SS of the next, one pitch apart in rt
        # (measured per span at the inlet plane). Shift PS by that pitch so both faces form a single blade.
        pitch = coords[2, 0, :, -1] - coords[2, 0, :, 0]  # (J,)
        plot_faces = {k: f.clone() for k, f in faces.items()}
        plot_faces["PS"][2] += pitch[None, :]

        both = np.concatenate([v.ravel() for v in actual.values()] + [v.ravel() for v in pred.values()])
        emax = max(np.abs(v).max() for v in err.values())
        rel = np.sqrt(sum((v**2).sum() for v in err.values()) / sum((v**2).sum() for v in actual.values()))
        print(f"blade-surface pressure rel L2 error: {rel:.4f}, max |err| {emax:.4g}")

        panels = [
            ("Actual pressure", actual, Normalize(both.min(), both.max()), "viridis"),
            ("Predicted pressure", pred, Normalize(both.min(), both.max()), "viridis"),
            ("Error (pred - actual)", err, TwoSlopeNorm(0, -emax, emax), "RdBu_r"),
        ]
        if a.plot == "2d":  # each face unwrapped: axial x across, span r up
            fig, axs = plt.subplots(2, 3, figsize=(17, 8.5), sharex=True, sharey=True, constrained_layout=True)
            for col, (title, data, norm, cmap) in enumerate(panels):
                for row, (k, name) in enumerate((("PS", "pressure side"), ("SS", "suction side"))):
                    ax = axs[row, col]
                    x, r, _ = faces[k].numpy()
                    ax.pcolormesh(x, r, data[k], norm=norm, cmap=cmap, shading="gouraud")
                    ax.set_title(f"{title} ({name})")
                    if row == 1:
                        ax.set_xlabel("x (axial)")
                    if col == 0:
                        ax.set_ylabel("r (hub to midspan)")
                fig.colorbar(cm.ScalarMappable(norm, cmap), ax=axs[:, col], shrink=0.9, label="p")
            fig.suptitle(f"Blade pressure, case {case}  (rel L2 {rel:.3f})", fontsize=14)
        else:
            views = [("pressure-side view", 125), ("suction-side view", -55)]  # opposite sides of the blade
            allp = torch.cat([f.flatten(1) for f in plot_faces.values()], dim=1).numpy()  # (3, n) as x, r, rt
            fig = plt.figure(figsize=(20, 13))
            for row, (view, azim) in enumerate(views):
                for col, (title, data, norm, cmap) in enumerate(panels):
                    ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
                    for k, f in plot_faces.items():
                        x, r, rt = f.numpy()
                        ax.plot_surface(x, rt, r, facecolors=plt.get_cmap(cmap)(norm(data[k])), shade=False,
                                        rstride=1, cstride=1, linewidth=0, antialiased=False)
                    fig.colorbar(cm.ScalarMappable(norm, cmap), ax=ax, shrink=0.6, pad=0.05, label="p")
                    ax.set_xlabel("x")
                    ax.set_ylabel("rt")
                    ax.set_zlabel("r")
                    ax.set_title(f"{title} ({view})")
                    ax.set_box_aspect(np.ptp(allp[[0, 2, 1]], axis=1))  # true proportions in (x, rt, r)
                    ax.view_init(elev=20, azim=azim)
            fig.suptitle(f"Blade pressure, case {case}  (rel L2 {rel:.3f})", y=0.97, fontsize=14)
            fig.subplots_adjust(left=0.02, right=0.96, bottom=0.04, top=0.9, wspace=0.25, hspace=0.3)
        return fig, rel

    rows = []
    for n, case in enumerate(cases, 1):
        target = Path(a.out_dir) / f"{case}.png" if a.out_dir else None
        if target and a.skip_existing and target.exists():
            continue
        fig, rel = plot_case(case)
        rows.append((case, rel))
        if target:
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=150)
            print(f"[{n}/{len(cases)}] saved {target}", flush=True)
        elif a.out:
            fig.savefig(a.out, dpi=150)
            print("saved", a.out)
        else:
            plt.show()
        plt.close(fig)
    if a.out_dir and rows:
        with open(Path(a.out_dir) / "errors.csv", "a", newline="") as fh:
            csv.writer(fh).writerows(rows)  # case, blade-surface pressure rel L2


if __name__ == "__main__":
    main()
