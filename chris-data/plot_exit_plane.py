"""Plot actual vs predicted vs error on a constant-axial-station plane downstream of the trailing edge,
as a 3D colour plot with the blade shown in grey for context (two views, like plot_blade_pressure.py).

The plane is a full pitch (rt) by span (r) slice at axial index I. Valid I runs from the trailing edge
(66) to the last usable station (84); the default is the midpoint.

    python plot_exit_plane.py --ckpt best.pt                      # p, first validation case, I = 75
    python plot_exit_plane.py --ckpt best.pt --var vx --i 84 --case g222 --out plane.png
    python plot_exit_plane.py --ckpt best.pt --var ske --i 84        # secondary kinetic energy
    python plot_exit_plane.py --ckpt best.pt --var yp                # local stagnation pressure loss coefficient
Get the checkpoint from Modal first:  modal volume get geotransolver-runs /geotransolver/best.pt .
"""
import argparse
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
from tensor_dataset import N_AXIAL_USABLE, CascadeDataset  # noqa: E402
from plot_loss_coefficient import losses, mass_avg, p0_from  # noqa: E402
from predict_cache import cached_predict_field  # noqa: E402
from surface_model import SurfaceModel, is_surface  # noqa: E402
from train_geotransolver import LE_I, TE_I, blade_surface, build_model  # noqa: E402


def ref_angle(f, rt, ref):
    """Mass-flux-averaged exit flow yaw (rad) at each span, shape (J, 1). `midspan` uses the last (midspan) row everywhere."""
    dx = np.abs(np.diff(rt, axis=1))  # pitchwise cell widths (J, K-1)
    trap = lambda y: (0.5 * (y[:, 1:] + y[:, :-1]) * dx).sum(axis=1)  # trapezoid over a pitch, per span
    w = f["ro"] * f["vx"]
    alpha = np.arctan2(trap(w * f["vt"]), trap(w * f["vx"]))
    return (np.full_like(alpha, alpha[-1]) if ref == "midspan" else alpha)[:, None]


def ske_field(f, alpha):
    """Secondary kinetic energy density 0.5*rho*V_sec^2: V_sec is the velocity minus its component along the
    reference exit direction (cos a, sin a) in the axial-tangential plane, i.e. spanwise vr plus the pitchwise
    deviation from that direction."""
    v_perp = -f["vx"] * np.sin(alpha) + f["vt"] * np.cos(alpha)
    return 0.5 * f["ro"] * (f["vr"] ** 2 + v_perp**2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=str(ROOT / "grid125_R23"))
    ap.add_argument("--case", default=None, help="case id (default: first validation case)")
    ap.add_argument("--var", default="p", help="variable to plot (in the checkpoint's variables), `ske` "
                    "(secondary kinetic energy) or `yp` (local stagnation pressure loss coefficient); "
                    "ske and yp need ro, vx, vr, vt, p")
    ap.add_argument("--ref", choices=["midspan", "local"], default="midspan",
                    help="SKE reference flow direction: mass-averaged yaw at midspan (default) or at each span")
    ap.add_argument("--i", type=int, default=None,
                    help=f"axial station, {TE_I} (trailing edge) to {N_AXIAL_USABLE - 1} (end of mesh). Default: the "
                         f"product plane ({N_AXIAL_USABLE - 1}) for ske/yp, else halfway ({(TE_I + N_AXIAL_USABLE - 1) // 2})")
    ap.add_argument("--plot", choices=["2d", "3d"], default="2d", help="2d flat maps (default) or 3d surface plots")
    ap.add_argument("--cache", default=None, help="directory to cache whole-mesh predictions (shared across plots)")
    ap.add_argument("--out", default=None, help="save to this file instead of showing")
    a = ap.parse_args()

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args, nm = ckpt["args"], ckpt["norm"]
    variables = args["variables"]
    surface = is_surface(ckpt)  # blade-surface + plane model: only predicts its own plane, any derived variable works
    if surface:
        sm = SurfaceModel(ckpt)
        if a.i is not None and a.i != sm.plane_i:
            sys.exit(f"this checkpoint only predicts the plane I={sm.plane_i}, not I={a.i}")
        a.i = sm.plane_i
    elif a.i is None:
        a.i = N_AXIAL_USABLE - 1 if a.var in ("ske", "yp") else (TE_I + N_AXIAL_USABLE - 1) // 2
    if not TE_I <= a.i <= N_AXIAL_USABLE - 1:
        sys.exit(f"--i must be between {TE_I} and {N_AXIAL_USABLE - 1}")
    need = ["ro", "vx", "vr", "vt", "p"] if a.var in ("ske", "yp") else [a.var]
    if not surface and any(v not in variables for v in need):
        sys.exit(f"{a.var!r} needs {need}, but checkpoint variables are {variables}")

    ds = CascadeDataset(a.data, variables=args["variables"])
    ids = [r["case"] for r in ds.rows]
    if a.case is None:  # train_geotransolver trains on evenly spread `picks`; val_idx indexes that list
        picks = torch.linspace(0, len(ds) - 1, len(ckpt["train_idx"]) + len(ckpt["val_idx"])).round().long().tolist()
        a.case = ids[picks[ckpt["val_idx"][0]]]
    s = ds[ids.index(a.case)]
    print("case", a.case, "params (sweep, lean, theta):", s["params"].tolist(), "| axial station", a.i)

    plane = s["coords"][:, a.i]  # (3, J, K): x, r, rt
    if surface:
        act_all, pred_all = sm.predict_case(s, ds.rows[ids.index(a.case)])
        act_f = {v: d["plane"] for v, d in act_all.items()}  # conserved + derived vx vr vt p M p0
        pred_f = {v: d["plane"] for v, d in pred_all.items()}
        if a.var not in act_f and a.var not in ("ske", "yp"):
            sys.exit(f"unknown variable {a.var!r}; choose from {sorted(act_f)} or ske, yp")
    else:
        model = build_model(args, len(args["variables"]))
        model.load_state_dict(ckpt["model"])
        model.eval()

        def norm_coords(c):
            return 2 * (c - nm["c_min"]) / (nm["c_max"] - nm["c_min"]).clamp_min(1e-8) - 1

        g = (2 * (s["params"] - nm["p_min"]) / (nm["p_max"] - nm["p_min"]).clamp_min(1e-8) - 1)[None, None]
        geo = norm_coords(blade_surface(s["coords"]))[None]
        # predict the whole mesh in training-like random chunks, then read off the plane
        full = cached_predict_field(a.cache, a.ckpt, a.case, model, norm_coords(s["coords"].flatten(1).T), g, geo,
                                    args["n_points"])  # (N, C)
        full = (full * nm["f_std"] + nm["f_mean"]).reshape(*s["coords"].shape[1:], -1)[a.i].numpy()  # (J, K, C)
        act_f = {v: s["fields"][variables.index(v), a.i].numpy() for v in variables}
        pred_f = {v: full[..., variables.index(v)] for v in variables}
    if a.var == "ske":
        label = "SKE"
        alpha = ref_angle(act_f, plane[2].numpy(), a.ref)  # same reference for actual and predicted
        actual, pred = ske_field(act_f, alpha), ske_field(pred_f, alpha)
        print(f"SKE reference yaw {np.degrees(alpha[:, 0]).min():.2f}..{np.degrees(alpha[:, 0]).max():.2f} deg "
              f"({a.ref}); plane-mean SKE actual {actual.mean():.4g}, predicted {pred.mean():.4g}")
    elif a.var == "yp":
        label = "Yp (local)"
        row = ds.rows[ids.index(a.case)]
        gamma, p01 = float(row["Gamma"]), float(row["pp_p0_inlet"])
        rt_p, r_p = plane[2].numpy(), plane[1].numpy()
        p02 = mass_avg(p0_from(act_f, gamma), act_f, rt_p, r_p)  # normalisation from the actual field,
        p2 = mass_avg(act_f["p"], act_f, rt_p, r_p)  # used for both so the maps are comparable
        actual, pred = ((p01 - p0_from(f, gamma)) / (p02 - p2) for f in (act_f, pred_f))
        (yp_a, _), (yp_p, _) = (losses(f, gamma, p01, rt_p, r_p) for f in (act_f, pred_f))
        print(f"Yp = (p01 - p0) / (p02 - p2), p01 {p01:.0f} p02 {p02:.0f} p2 {p2:.0f}; "
              f"mass-averaged Yp actual {yp_a:.4f}, predicted {yp_p:.4f}")
    else:
        label = a.var
        actual, pred = act_f[a.var], pred_f[a.var]
    err = pred - actual
    rel = np.linalg.norm(err) / np.linalg.norm(actual)
    # relative to the variation across the plane, which is what matters when the mean dominates
    rel_dev = np.linalg.norm(err - err.mean()) / np.linalg.norm(actual - actual.mean())
    print(f"plane rel L2 {rel:.4f}; rel L2 of variation about the plane mean {rel_dev:.4f}; max |err| {np.abs(err).max():.4g}")

    both = np.concatenate([actual.ravel(), pred.ravel()])
    emax = np.abs(err).max()
    panels = [
        (f"Actual {label}", actual, Normalize(both.min(), both.max()), "viridis"),
        (f"Predicted {label}", pred, Normalize(both.min(), both.max()), "viridis"),
        ("Error (pred - actual)", err, TwoSlopeNorm(0, -emax, emax), "RdBu_r"),
    ]
    # Blade for context: same construction as plot_blade_pressure.py (PS shifted by one pitch so the faces close up)
    c4 = s["coords"]
    pitch = c4[2, 0, :, -1] - c4[2, 0, :, 0]
    blade = {"PS": c4[:, LE_I : TE_I + 1, :, 0].clone(), "SS": c4[:, LE_I : TE_I + 1, :, -1].clone()}
    blade["PS"][2] += pitch[None, :]
    allp = torch.cat([plane.flatten(1)] + [f.flatten(1) for f in blade.values()], dim=1).numpy()  # (3, n): x, r, rt

    xp, rp, rtp = plane.numpy()
    if a.plot == "2d":  # flat pitch (rt) by span (r) map
        fig, axs = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
        for ax, (title, z, norm, cmap) in zip(axs, panels):
            ax.pcolormesh(rtp, rp, z, norm=norm, cmap=cmap, shading="gouraud")
            fig.colorbar(cm.ScalarMappable(norm, cmap), ax=ax, shrink=0.85, label=label)
            ax.set_xlabel("rt (pitchwise)")
            ax.set_ylabel("r (hub to midspan)")
            ax.set_title(title)
            ax.set_aspect("equal")
        fig.suptitle(f"{label} on axial plane I={a.i} (x = {xp.mean():.3f}), case {a.case}  (rel L2 {rel:.3f})",
                     fontsize=14)
    else:
        views = [("upstream view", -55), ("downstream view", 125)]  # opposite sides, as in the blade plots
        fig = plt.figure(figsize=(20, 13))
        for row, (view, azim) in enumerate(views):
            for col, (title, z, norm, cmap) in enumerate(panels):
                ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
                for f in blade.values():
                    x, r, rt = f.numpy()
                    ax.plot_surface(x, rt, r, color="0.55", alpha=0.6, shade=False, linewidth=0, antialiased=False)
                ax.plot_surface(xp, rtp, rp, facecolors=plt.get_cmap(cmap)(norm(z)), shade=False,
                                rstride=1, cstride=1, linewidth=0, antialiased=False)
                fig.colorbar(cm.ScalarMappable(norm, cmap), ax=ax, shrink=0.6, pad=0.05, label=label)
                ax.set_xlabel("x")
                ax.set_ylabel("rt")
                ax.set_zlabel("r")
                ax.set_title(f"{title} ({view})")
                ax.set_box_aspect(np.ptp(allp[[0, 2, 1]], axis=1))  # true proportions in (x, rt, r)
                ax.view_init(elev=20, azim=azim)
        fig.suptitle(f"{label} on axial plane I={a.i} (x = {xp.mean():.3f}), case {a.case}  (rel L2 {rel:.3f})",
                     y=0.97, fontsize=14)
        fig.subplots_adjust(left=0.02, right=0.96, bottom=0.04, top=0.9, wspace=0.25, hspace=0.3)
    if a.out:
        fig.savefig(a.out, dpi=150)
        print("saved", a.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
