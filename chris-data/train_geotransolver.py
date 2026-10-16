"""Train GeoTransolver as a surrogate: (mesh coords, sweep/lean/theta) -> flow field.

Per case, tokens are mesh points:
    local_embedding = [x, r, rt (normalised), sweep, lean, theta (normalised)]   (B, N, 6)
    geometry        = blade-surface points [x, r, rt] (normalised)               (B, N_blade, 3)
                      = pressure (K=0) + suction (K=60) faces for I = LE..TE, all spans (~7.9k pts)
    global_embedding= [sweep, lean, theta (normalised)]                          (B, 1, 3)
    target          = flow variables, standardised per channel                   (B, N, C)
Training uses a random subset of `--n_points` mesh points per case each step (a full case is
~378k points); the validation cases are scored on a fixed random subset of `--eval_points`.
`--params` chooses where sweep/lean/theta enter: `both` (default; per point and as the global embedding), `global`
(only the global embedding; the local embedding is just [x, r, rt], (B, N, 3)) or `none` (not an input at all, so a case is
known only through the blade-surface geometry context; theta does not change that geometry, so its effect cannot be learned).

Run from anywhere:  python train_geotransolver.py --epochs 100
"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "chris-data"))
sys.path.insert(0, str(ROOT))
from geotransolver_torch import GeoTransolver  # noqa: E402
from tensor_dataset import CascadeDataset  # noqa: E402


LE_I, TE_I = 13, 66  # 0-based leading / trailing edge stations (README); K=0 PS, K=60 SS in between


def blade_surface(coords):
    """coords (3, I, J, K) -> (N_blade, 3) points on the blade (PS + SS faces, LE..TE, all spans)."""
    ps = coords[:, LE_I : TE_I + 1, :, 0]
    ss = coords[:, LE_I : TE_I + 1, :, -1]
    return torch.cat([ps.flatten(1), ss.flatten(1)], dim=1).T.contiguous()


def params_mode(a):
    """Where sweep/lean/theta enter for an args dict: 'both', 'global' or 'none' (older checkpoints: 'both')."""
    return a.get("params") or ("none" if a.get("no_params") else "both")


def build_model(a, n_ch):
    """GeoTransolver for an args dict `a` (CLI args or a checkpoint's), honouring `params` (see params_mode)."""
    mode = params_mode(a)
    model = GeoTransolver(
        functional_dim=6 if mode == "both" else 3, out_dim=n_ch, geometry_dim=3,
        global_dim=3 if mode in ("both", "global") else None,
        n_layers=a["n_layers"], n_hidden=a["n_hidden"], n_head=a["n_head"], slice_num=a["slice_num"],
    )
    model.params_mode = mode
    return model


def model_inputs(model, c, g):
    """(local_embedding, global_embedding) for normalised coords c (B, n, 3) and normalised params g (B, 1, 3)."""
    if model.params_mode == "both":
        return torch.cat([c, g.expand(-1, c.shape[1], -1)], dim=-1), g
    return c, (g if model.params_mode == "global" else None)


@torch.no_grad()
def predict_field(model, coords_n, g, geo, chunk=16384):
    """Predict every mesh point of one case -> (N, C), in standardised units.

    The slice attention pools over all input points, so a point's output depends on which other points are in
    the batch. Predicting a lone subset (a wall, a plane) is out of distribution and gives badly wrong values; so
    feed random chunks of the whole mesh, as in training, and pick the points you want out of the result.
    coords_n (N, 3) normalised; g (1, 1, 3) normalised params; geo (1, N_blade, 3) normalised blade points.
    """
    n = coords_n.shape[0]
    out = None
    for idx in torch.tensor_split(torch.randperm(n, generator=torch.Generator().manual_seed(0)), -(-n // chunk)):
        c = coords_n[idx][None]
        local, gl = model_inputs(model, c, g)
        pred = model(local, geometry=geo, global_embedding=gl)[0]
        if out is None:
            out = torch.empty(n, pred.shape[1])
        out[idx] = pred.cpu()
    return out


def near_wall_indices(grid, layers):
    """Flat point indices of the first/last `layers` pitchwise (K) layers: the PS and SS walls over all I, J."""
    n_i, n_j, n_k = grid
    k = torch.cat([torch.arange(layers), torch.arange(n_k - layers, n_k)])
    return ((torch.arange(n_i * n_j) * n_k)[:, None] + k[None, :]).flatten()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "chris-data" / "grid125_R23"))
    p.add_argument("--variables", nargs="+", default=["ro", "vx", "vr", "vt", "p"])
    p.add_argument("--params", choices=["both", "global", "none"], default="both",
                   help="where sweep/lean/theta enter: per point and global (both), only the global embedding, or nowhere")
    p.add_argument("--no_params", action="store_true", help="same as --params none")
    p.add_argument("--max_cases", type=int, default=None, help="use only the first N cases (debugging)")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--n_points", type=int, default=16384, help="points sampled per case per training step")
    p.add_argument("--eval_points", type=int, default=32768, help="points per val case (fixed subset)")
    p.add_argument("--wall_frac", type=float, default=0.0,
                   help="fraction of --n_points drawn from the near-wall layers each step (0 = uniform sampling)")
    p.add_argument("--wall_layers", type=int, default=3, help="pitchwise layers next to each wall used by --wall_frac")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_hidden", type=int, default=128)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--slice_num", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "runs" / "geotransolver"))
    return p.parse_args()


def load_all(args):
    """Load every case into memory as flat per-case tensors (N = I*J*K points)."""
    ds = CascadeDataset(args.data, variables=args.variables)
    n = len(ds) if args.max_cases is None else min(args.max_cases, len(ds))
    picks = torch.linspace(0, len(ds) - 1, n).round().long().tolist()  # evenly spread over the design
    fields, coords, blades, params = [], [], [], []
    for i in picks:
        s = ds[i]
        fields.append(s["fields"].flatten(1).T.contiguous())  # (N, C)
        coords.append(s["coords"].flatten(1).T.contiguous())  # (N, 3)
        blades.append(blade_surface(s["coords"]))  # (N_blade, 3)
        params.append(s["params"])
    print(f"loaded {n} cases, {fields[0].shape[0]} points + {blades[0].shape[0]} blade points each, "
          f"channels {args.variables}")
    return fields, coords, blades, torch.stack(params), tuple(s["fields"].shape[1:])


def main():
    args = parse_args()
    if args.no_params:
        args.params = "none"
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fields, coords, blades, params, grid = load_all(args)
    n_cases, n_pts, n_ch = len(fields), fields[0].shape[0], fields[0].shape[1]

    # Split by case so validation designs are genuinely unseen.
    perm = torch.randperm(n_cases)
    n_val = max(1, int(round(args.val_frac * n_cases))) if n_cases > 1 else 0
    val_idx, train_idx = perm[:n_val].tolist(), perm[n_val:].tolist()
    print(f"train {len(train_idx)} cases, val {len(val_idx)} cases, device {device}")

    # Normalisation statistics come from training cases only.
    tr_fields = torch.cat([fields[i] for i in train_idx])
    f_mean, f_std = tr_fields.mean(0), tr_fields.std(0).clamp_min(1e-8)
    del tr_fields
    all_coords = torch.cat(coords)
    c_min, c_max = all_coords.min(0).values, all_coords.max(0).values
    del all_coords
    p_min, p_max = params.min(0).values, params.max(0).values

    def norm_coords(c):
        return 2 * (c - c_min) / (c_max - c_min).clamp_min(1e-8) - 1

    def norm_params(p):
        return 2 * (p - p_min) / (p_max - p_min).clamp_min(1e-8) - 1

    def make_batch(idx, pts):
        """idx: list of case indices; pts: (n,) point indices (shared across the batch)."""
        c = torch.stack([norm_coords(coords[i][pts]) for i in idx])  # (B, n, 3)
        g = norm_params(params[idx])[:, None, :]  # (B, 1, 3)
        local, gl = model_inputs(model, c, g)  # (B, n, 6) and (B, 1, 3), or (B, n, 3) and None with --no_params
        y = torch.stack([(fields[i][pts] - f_mean) / f_std for i in idx])  # (B, n, C)
        geo = torch.stack([norm_coords(blades[i]) for i in idx])  # (B, N_blade, 3)
        return local.to(device), geo.to(device), None if gl is None else gl.to(device), y.to(device)

    model = build_model(vars(args), n_ch).to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_idx) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)
    amp = device == "cuda"

    g = torch.Generator().manual_seed(args.seed)
    val_pts = torch.randperm(n_pts, generator=g)[: min(args.eval_points, n_pts)]
    train_wall = near_wall_indices(grid, args.wall_layers)
    # wall points that fall inside the random validation subset (scored in-context, see predict_field)
    on_wall = ((val_pts % grid[2] == 0) | (val_pts % grid[2] == grid[2] - 1)).to(device)
    n_wall = int(args.wall_frac * args.n_points)

    @torch.no_grad()
    def evaluate():
        model.eval()
        mse, rel, wall = [], [], []
        for i in val_idx:
            local, geo, gl, y = make_batch([i], val_pts)
            pred = model(local, geometry=geo, global_embedding=gl)
            mse.append(((pred - y) ** 2).mean(dim=(0, 1)))  # per channel, normalised units
            rel.append((pred - y).norm(dim=(0, 1)) / y.norm(dim=(0, 1)))
            wall.append(((pred - y)[:, on_wall] ** 2).mean())
        return torch.stack(mse).mean(0), torch.stack(rel).mean(0), torch.stack(wall).mean()

    best = float("inf")
    hist = out / "history.csv"  # one row per epoch, for plot_training_curves.py
    hist.write_text("epoch,train_mse,val_mse,val_wall_mse," + ",".join(f"val_relL2_{v}" for v in args.variables) + "\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = [train_idx[j] for j in torch.randperm(len(train_idx)).tolist()]
        total, t0 = 0.0, time.time()
        for s in range(steps_per_epoch):
            idx = order[s * args.batch_size : (s + 1) * args.batch_size]
            pts = torch.randperm(n_pts)[: args.n_points - n_wall]
            if n_wall:
                pts = torch.cat([pts, train_wall[torch.randint(len(train_wall), (n_wall,))]])
            local, geo, gl, y = make_batch(idx, pts)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                pred = model(local, geometry=geo, global_embedding=gl)
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
            mse, rel, wall = evaluate()
            val = mse.mean().item()
            row += [val, wall.item(), *rel.tolist()]
            msg += f" | val_mse {val:.4e} | val_relL2 " + " ".join(f"{v:.3f}" for v in rel.tolist())
            msg += f" | val_wall_mse {wall.item():.4e}"
            if val < best:
                best = val
                torch.save({
                    "model": model.state_dict(), "args": vars(args), "epoch": epoch,
                    "norm": {"f_mean": f_mean, "f_std": f_std, "c_min": c_min, "c_max": c_max,
                             "p_min": p_min, "p_max": p_max},
                    "val_idx": val_idx, "train_idx": train_idx,
                }, out / "best.pt")
        print(msg, flush=True)
        with open(hist, "a") as fh:
            fh.write(",".join(f"{v:.6g}" for v in row) + "\n")


if __name__ == "__main__":
    main()
