"""Train GeoTransolver on the blade surface + one downstream plane: (point, normal, inlet conditions) -> flow.

Per case, the tokens are only the points where we want the flow, not the whole mesh:
    local_embedding = [x, r, rt (normalised), nx, nr, nrt]                          (B, N, 6)
        blade points : pressure (K=0) + suction (K=last) faces, LE..TE, all spans, with unit surface normals
                       pointing into the flow
        plane points : the full pitch x span plane at axial index --plane_i (default 75), with the unit normal
                       of the mesh surface at each point, pointing downstream
    geometry        = blade-surface points [x, r, rt] (normalised)                  (B, N_blade, 3)
    global_embedding= case-table columns --global_cols (normalised): by default the inlet flow angle A1, the
                      exit Mach number M2 and theta_unit                              (B, 1, G)
    target          = the stored conserved fields ro rovx rovr rorvt roe at those points, standardised per
                      channel for training (undo with the saved f_mean / f_std to get the table values back)   (B, N, 5)
Every step uses the same kind of point set (all blade + plane points, ~12k per case), so the attention context at
inference matches training. Coordinates are in the stored (x, r, rt) frame, treated as locally Cartesian.

In this dataset only sweep, lean and theta vary; the other table columns are constant, so most global inputs carry
no information here. Sweep and lean reach the model through the blade geometry; theta (which does not change the
grid; it looks like the inlet boundary-layer parameter) through the global embedding.

    python train_geotransolver_surface.py --data grid125_R23 --epochs 300
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from geotransolver_torch import GeoTransolver  # noqa: E402
from tensor_dataset import CONSERVED, CascadeDataset  # noqa: E402
from train_geotransolver import LE_I, TE_I  # noqa: E402

def face_normals(face, inner):
    """face (3, A, B): points on a mesh surface; inner (3, A, B): the adjacent grid layer on the side the normals
    should point to (the flow for a blade face, downstream for the plane). Returns unit normals (3, A, B)."""
    n = np.cross(np.gradient(face, axis=1), np.gradient(face, axis=2), axis=0)
    n /= np.linalg.norm(n, axis=0, keepdims=True).clip(1e-12)
    side = np.sign((n * (inner - face)).sum(axis=0, keepdims=True))  # which way the interior lies
    return n * np.where(side == 0, 1.0, side)


def case_points(coords, fields, plane_i):
    """coords (3, I, J, K), fields (C, I, J, K) tensors -> pts (N, 3), normals (N, 3), y (N, C), n_blade.
    The first n_blade rows are the blade surface (PS then SS), the rest the plane."""
    c, f = coords.numpy(), fields.numpy()
    le_te = slice(LE_I, TE_I + 1)
    pts, nrm, ys = [], [], []
    for k, k_in in ((0, 1), (-1, -2)):  # pressure face, suction face (and the layer next to each)
        face = c[:, le_te, :, k]
        pts.append(face.reshape(3, -1).T)
        nrm.append(face_normals(face, c[:, le_te, :, k_in]).reshape(3, -1).T)
        ys.append(f[:, le_te, :, k].reshape(f.shape[0], -1).T)
    n_blade = sum(len(p) for p in pts)
    plane = c[:, plane_i]  # (3, J, K)
    downstream = c[:, plane_i + 1] if plane_i + 1 < c.shape[1] else 2 * plane - c[:, plane_i - 1]
    pts.append(plane.reshape(3, -1).T)
    nrm.append(face_normals(plane, downstream).reshape(3, -1).T)
    ys.append(f[:, plane_i].reshape(f.shape[0], -1).T)
    cat = lambda xs: torch.from_numpy(np.concatenate(xs).astype(np.float32))  # noqa: E731
    return cat(pts), cat(nrm), cat(ys), n_blade


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "grid125_R23"))
    p.add_argument("--variables", nargs="+", default=list(CONSERVED), help="fields to predict (table values)")
    p.add_argument("--global_cols", nargs="+", default=["A1", "M2", "theta_unit"],
                   help="cases.csv columns used as global inputs")
    p.add_argument("--plane_i", type=int, default=75, help="axial index of the plane (TE is 66, last usable is 84)")
    p.add_argument("--max_cases", type=int, default=None, help="use only the first N cases (debugging)")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_points", type=int, default=0, help="random points per case per step (0 = all of them)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_hidden", type=int, default=128)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--slice_num", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "runs" / "geotransolver_surface"))
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = CascadeDataset(args.data, variables=args.variables)
    missing = [c for c in args.global_cols if c not in ds.rows[0]]
    if missing:
        sys.exit(f"--global_cols {missing} not in cases.csv columns")
    n = len(ds) if args.max_cases is None else min(args.max_cases, len(ds))
    picks = torch.linspace(0, len(ds) - 1, n).round().long().tolist()  # evenly spread over the design
    pts, nrm, ys, glob, n_blade = [], [], [], [], None
    for i in picks:
        s = ds[i]
        p_, n_, y_, nb = case_points(s["coords"], s["fields"], args.plane_i)
        pts.append(p_), nrm.append(n_), ys.append(y_)
        glob.append([float(ds.rows[i][c]) for c in args.global_cols])
        n_blade = nb if n_blade is None else n_blade
    glob = torch.tensor(glob)
    n_pts, n_ch = pts[0].shape[0], ys[0].shape[1]
    print(f"loaded {n} cases: {n_blade} blade + {n_pts - n_blade} plane (I={args.plane_i}) points each, "
          f"channels {args.variables}, global {args.global_cols}")

    # Split by case so validation designs are genuinely unseen.
    perm = torch.randperm(n)
    n_val = max(1, int(round(args.val_frac * n))) if n > 1 else 0
    val_idx, train_idx = perm[:n_val].tolist(), perm[n_val:].tolist()
    print(f"train {len(train_idx)} cases, val {len(val_idx)} cases, device {device}")

    # Normalisation statistics (fields from training cases only).
    tr_y = torch.cat([ys[i] for i in train_idx])
    f_mean, f_std = tr_y.mean(0), tr_y.std(0).clamp_min(1e-8)
    all_pts = torch.cat(pts)
    c_min, c_max = all_pts.min(0).values, all_pts.max(0).values
    g_min, g_max = glob.min(0).values, glob.max(0).values  # constant columns map to 0
    del tr_y, all_pts

    norm_c = lambda c: 2 * (c - c_min) / (c_max - c_min).clamp_min(1e-8) - 1  # noqa: E731
    local = [torch.cat([norm_c(pts[i]), nrm[i]], dim=-1) for i in range(n)]  # (N, 6)
    geo = [norm_c(pts[i][:n_blade]) for i in range(n)]  # (N_blade, 3)
    glb = (2 * (glob - g_min) / (g_max - g_min).clamp_min(1e-8) - 1)[:, None, :]  # (n, 1, G)
    tgt = [(ys[i] - f_mean) / f_std for i in range(n)]  # (N, C)

    def make_batch(idx, sub=None):
        """idx: list of case indices; sub: optional (m,) point indices shared across the batch."""
        sel = (lambda t: t) if sub is None else (lambda t: t[sub])
        x = torch.stack([sel(local[i]) for i in idx])
        y = torch.stack([sel(tgt[i]) for i in idx])
        return (x.to(device), torch.stack([geo[i] for i in idx]).to(device),
                glb[idx].to(device), y.to(device))

    model = GeoTransolver(
        functional_dim=6, out_dim=n_ch, geometry_dim=3, global_dim=len(args.global_cols),
        n_layers=args.n_layers, n_hidden=args.n_hidden, n_head=args.n_head, slice_num=args.slice_num,
    ).to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_idx) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)
    amp = device == "cuda"

    @torch.no_grad()
    def evaluate():
        model.eval()
        mse, rel, blade, plane = [], [], [], []
        for i in val_idx:
            x, gm, gl, y = make_batch([i])
            err = model(x, geometry=gm, global_embedding=gl) - y
            mse.append((err**2).mean(dim=(0, 1)))  # per channel, standardised units
            rel.append(err.norm(dim=(0, 1)) / y.norm(dim=(0, 1)))
            blade.append((err[:, :n_blade] ** 2).mean())
            plane.append((err[:, n_blade:] ** 2).mean())
        return torch.stack(mse).mean(0), torch.stack(rel).mean(0), torch.stack(blade).mean(), torch.stack(plane).mean()

    best = float("inf")
    hist = out / "history.csv"  # one row per epoch, for plot_training_curves.py
    hist.write_text("epoch,train_mse,val_mse,val_blade_mse,val_plane_mse,"
                    + ",".join(f"val_relL2_{v}" for v in args.variables) + "\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = [train_idx[j] for j in torch.randperm(len(train_idx)).tolist()]
        total, t0 = 0.0, time.time()
        for s in range(steps_per_epoch):
            idx = order[s * args.batch_size : (s + 1) * args.batch_size]
            sub = torch.randperm(n_pts)[: args.n_points] if 0 < args.n_points < n_pts else None
            x, gm, gl, y = make_batch(idx, sub)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                pred = model(x, geometry=gm, global_embedding=gl)
            loss = torch.nn.functional.mse_loss(pred.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        msg = f"epoch {epoch:4d} train_mse {total / steps_per_epoch:.4e} ({time.time() - t0:.0f}s)"
        row = [epoch, total / steps_per_epoch]
        if val_idx:
            mse, rel, blade, plane = evaluate()
            val = mse.mean().item()
            row += [val, blade.item(), plane.item(), *rel.tolist()]
            msg += (f" | val_mse {val:.4e} (blade {blade.item():.3e} plane {plane.item():.3e})"
                    f" | val_relL2 " + " ".join(f"{v:.3f}" for v in rel.tolist()))
            if val < best:
                best = val
                torch.save({
                    "model": model.state_dict(), "args": vars(args), "epoch": epoch, "kind": "surface_plane",
                    "norm": {"f_mean": f_mean, "f_std": f_std, "c_min": c_min, "c_max": c_max,
                             "g_min": g_min, "g_max": g_max},
                    "val_idx": val_idx, "train_idx": train_idx, "picks": picks, "n_blade": n_blade,
                }, out / "best.pt")
        print(msg, flush=True)
        with open(hist, "a") as fh:
            fh.write(",".join(f"{v:.6g}" for v in row) + "\n")


if __name__ == "__main__":
    main()
