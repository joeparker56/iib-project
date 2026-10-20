"""Train GeoTransolver on the grid125_R23 lean/sweep cascade, following the recipe of
NVIDIA's rotor37_geotransolver example: training-only normalisation, a held-out
validation split, zero-initialised outputs, an explicit QoI loss, and AdamW with a
cosine schedule and gradient clipping.

Overview of the pipeline
------------------------
1. prepare():    load all 125 CFD cases into memory and keep only the points we train on:
                 the blade pressure and suction surfaces (LE to TE, every span station)
                 and the exit / product plane at I=84 (~12.3k points per case instead of
                 the full ~378k). Split into train / validation and standardise everything
                 using statistics from the training cases only.
2. make_batch(): each optimisation step picks a few cases and, by default, uses every
                 selected point (optionally a random subset of n_query of them).
3. forward():    the model sees, per case, the query points, the wall geometry and
                 the design parameters, and predicts the 5 conserved variables at
                 every query point plus the case-level QoIs (e.g. p0 loss).
4. train():      AdamW + cosine learning-rate decay, loss = field MSE + w_qoi * QoI MSE.
5. validate():   predict every selected point of each held-out case and report errors
                 in physical units, separately for the blade surface and the exit plane.
"""
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Make the repo root importable (for `models/`) regardless of where the script is run from
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.geotransolver import GeoTransolver

# Datasets in the same format (cases.csv + fields/ + grids/), selected with cfg["dataset"].
#   path:   folder under datasets/ (on Modal: /data/<name> in the geotransolver-data volume)
#   params: the cases.csv inputs that vary in that dataset, fed to the model as its design inputs.
#           Only varying inputs can be used: a constant one has zero std and can't be normalised.
DATASETS = {
    "grid125_R23": dict(  # 125 cases: 5 sweep x 5 lean x 5 BL thickness, everything else fixed
        path="chris-surrogate-lean-sweep/grid125_R23",
        params=("sweep_w_deg", "lean_w_deg", "theta_unit"),
    ),
    "conv1000_c4v2": dict(  # 1000 converged cases of a 14-input Sobol design
        path="chris-cfd-outputs/conv1000_c4v2",
        params=("A1", "A2", "M2", "Mpeak", "lpeak", "Mless_rat", "Mps_rat", "FL", "Gamma", "Zte",
                "theta_unit", "lean_w_deg", "sweep_w_deg", "log2_AR"),
    ),
}
CONSERVED = ("ro", "rovx", "rovr", "rorvt", "roe")  # flow variables stored in fields/<case>.npz
N_AXIAL = 85     # keep I = 0..84: the last 2 stations are corrupted by the exit boundary condition (see README)
LE, TE = 13, 66  # 0-based axial stations of the leading / trailing edge
EXIT_I = 84      # 0-based axial station of the product plane, where the pp_* QoIs are evaluated
RESULTS = ROOT / "results"  # each training run gets its own numbered folder in here

CFG = dict(
    run_prefix="geotransolver_surface",  # runs are saved as results/<run_prefix>_01, _02, ...
    dataset="grid125_R23",  # a key of DATASETS
    params=None,    # design inputs; None = the dataset's default list (filled in when training starts)
    seed=42,        # fixes the train/val split, model initialisation and data order
    n_val=15,       # cases held out for validation (the rest are used for training)
    epochs=300,     # one epoch = one pass over every training case
    batch_size=4,   # cases per optimisation step
    lr=1e-3,        # peak learning rate (start of the cosine schedule)
    min_lr=1e-5,    # learning rate the cosine schedule decays to by the last epoch
    weight_decay=1e-4,  # AdamW's decoupled L2 penalty on the weights
    clip=1.0,       # maximum gradient norm; larger gradients are rescaled down to this
    n_query=None,  # points sampled per case per step; None = all ~12.3k blade + exit-plane points
    n_geom=2048,   # wall points in the geometry context
    qoi=("pp_p0_loss", "pp_exit_yaw_deg", "pp_exit_mach"), #quantities of interest
    w_qoi=4.0,     # the example weights its global outputs 4x
    val_every=25,   # run validation (and save a checkpoint) every this many epochs
    model=dict(     # GeoTransolver hyperparameters, saved with each run so changes are tracked
        hidden=128,             # width of each point token
        heads=4,                # attention heads (hidden must divide by heads)
        slices=16,              # learned "physics slices" each point is softly assigned to
        layers=4,               # number of GALE blocks
        geo_hidden=32,          # width of each ball-query geometry embedding
        radii=(0.1, 0.4, 1.0),  # ball-query radii in normalised coordinates (multi-scale)
        k=8,                    # max neighbours per ball query
    ),
)


def resolve(cfg):
    """A copy of cfg with the dataset's default params filled in, so the saved config is explicit."""
    if cfg["dataset"] not in DATASETS:
        raise ValueError(f"unknown dataset {cfg['dataset']!r}; choose from {list(DATASETS)}")
    cfg = dict(cfg)
    if cfg["params"] is None:
        cfg["params"] = DATASETS[cfg["dataset"]]["params"]
    return cfg


def data_dir(cfg):
    """Local folder of the run's dataset."""
    return ROOT / "datasets" / DATASETS[cfg["dataset"]]["path"]


def describe(cfg):
    """One-line summary of a run's settings, used to label figures."""
    m = cfg["model"]
    return (f"{cfg['dataset']} ({len(cfg['params'])} inputs) | hidden {m['hidden']}, {m['layers']} layers, "
            f"{m['slices']} slices, radii {tuple(m['radii'])} | {cfg['epochs']} epochs, lr {cfg['lr']:g}, "
            f"batch {cfg['batch_size']}, w_qoi {cfg['w_qoi']:g}, seed {cfg['seed']}")


def new_run_dir(results, prefix):
    """Create and return the next free numbered run folder, e.g. results/<prefix>_03."""
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    taken = [int(m.group(1)) for d in results.iterdir()
             if (m := re.fullmatch(rf"{re.escape(prefix)}_(\d+)", d.name))]
    run = results / f"{prefix}_{max(taken, default=0) + 1:02d}"
    run.mkdir()
    return run


def wall_points(coords, n):
    """n points on the walls: pressure (K=0) and suction (K=-1) surfaces from LE to TE, and the hub (J=0)."""
    # coords has shape (3, I, J, K): channel 0/1/2 = x, r, rt; I axial, J hub->midspan, K pitchwise.
    # In the blade passage, K=0 is the pressure surface and K=-1 the suction surface, but only
    # between the leading and trailing edge (upstream/downstream those K-faces are periodic
    # boundaries, not walls). J=0 is the hub, which is a wall along the whole axial length.
    surfaces = (coords[:, LE:TE + 1, :, 0], coords[:, LE:TE + 1, :, -1], coords[:, :, 0, :])
    # Flatten each surface to (3, n_points), join them, and transpose to (M, 3): one row per point.
    wall = torch.cat([s.reshape(3, -1) for s in surfaces], dim=1).T
    # Keep n evenly strided points. Because the grid is ordered consistently across cases,
    # the same indices pick out the "same" wall points on every geometry.
    return wall[torch.linspace(0, len(wall) - 1, n).long()]


def blade_and_exit(t):
    """Select the training points from a (C, I, J, K) grid tensor, returning (n_points, C).

    Works for both coords (C=3) and fields (C=5), so the same rows of each line up.
    Rows are ordered blade first, then exit plane:
        pressure surface  K=0,  I=LE..TE, all J   (54 x 73 = 3942 points)
        suction surface   K=-1, I=LE..TE, all J   (3942 points)
        exit plane        I=EXIT_I, all J and K   (73 x 61 = 4453 points)
    """
    parts = (t[:, LE:TE + 1, :, 0], t[:, LE:TE + 1, :, -1], t[:, EXIT_I])
    return torch.cat([s.reshape(len(t), -1) for s in parts], dim=1).T


def read_cases(data):
    """The rows of cases.csv: one dict per case (case id, grid file, inputs, QoIs), values as strings."""
    with open(Path(data) / "cases.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def load_case(row, data, params):
    """Load one case from its cases.csv row; params are the cases.csv input columns to use.

    fields: (5, I, J, K) conserved variables; coords: (3, I, J, K) = x, r, rt.
    Axes are I axial (inlet -> exit), J hub -> midspan, K pitchwise; only I < N_AXIAL is kept.
    """
    data = Path(data)
    with np.load(data / "fields" / f"{row['case']}.npz") as z:
        fields = np.stack([z[v][:N_AXIAL] for v in CONSERVED]).astype(np.float32)
    with np.load(data / "grids" / row["grid"]) as g:
        coords = np.stack([g[k][:N_AXIAL] for k in ("x", "r", "rt")]).astype(np.float32)
    return {
        "row": row,
        "fields": torch.from_numpy(fields),
        "coords": torch.from_numpy(coords),
        "params": torch.tensor([float(row[k]) for k in params]),
        "gamma": float(row["Gamma"]),
    }


def primitives(f, r, gamma):
    """Velocities, static pressure p and stagnation pressure p0 from the conserved variables in f
    (a dict of arrays keyed by CONSERVED, physical units); r is the radius at the same points."""
    ro = f["ro"]
    vx, vr, vt = f["rovx"] / ro, f["rovr"] / ro, f["rorvt"] / (ro * r)  # rorvt = rho * r * v_theta
    v2 = vx**2 + vr**2 + vt**2
    p = (gamma - 1) * (f["roe"] - 0.5 * ro * v2)  # roe = rho * total energy per unit mass
    mach2 = v2 / (gamma * p / ro)
    p0 = p * (1 + 0.5 * (gamma - 1) * mach2) ** (gamma / (gamma - 1))
    return {"vx": vx, "vr": vr, "vt": vt, "p": p, "p0": p0}


def split(n_cases, cfg):
    """Random but reproducible (train, val) case indices into the dataset's row order.

    A dedicated Generator means the split depends only on cfg["seed"], not on any other
    random numbers drawn earlier in the program.
    """
    perm = torch.randperm(n_cases, generator=torch.Generator().manual_seed(cfg["seed"])).tolist()
    return perm[cfg["n_val"]:], perm[:cfg["n_val"]]


def encode_case(s, cfg):
    """Turn one case from load_case() into the (unnormalised) blade + exit-plane point cloud."""
    return {
        "case": s["row"]["case"],
        "x": blade_and_exit(s["coords"]),  # (N, 3), N ~ 12.3k
        "geom": wall_points(s["coords"], cfg["n_geom"]),  # (n_geom, 3) wall point cloud
        "y": blade_and_exit(s["fields"]),  # (N, 5), same rows as x
        "p": s["params"],               # (n_params,) design inputs, e.g. sweep, lean, theta_unit
        "q": torch.tensor([float(s["row"][k]) for k in cfg["qoi"]]),  # (n_qoi,) product-plane QoIs
        # the first n_blade rows are blade-surface points, the rest the exit plane
        "n_blade": 2 * (TE - LE + 1) * s["coords"].shape[2],
    }


def normalise(c, stats):
    """Standardise a case in place with the (training-set) stats from prepare()."""
    # Broadcasting means a (3,) or (5,) mean/std is applied to every row of an (N, 3) or (N, 5) tensor.
    for key, (m, s) in stats.items():
        c[key] = (c[key] - m) / s
    # Wall points are coordinates too, so they use the coordinate transform
    c["geom"] = (c["geom"] - stats["x"][0]) / stats["x"][1]
    return c


def prepare(cfg, data):
    """Load every case, split, and normalise with training-set statistics only."""
    # --- 1. Load every case and keep only the blade-surface and exit-plane points ----------
    # Each case is (5, 85, 73, 61) fields + (3, 85, 73, 61) coords before selecting points.
    rows = read_cases(data)
    cases = []
    for i, row in enumerate(rows):
        cases.append(encode_case(load_case(row, data, cfg["params"]), cfg))
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f"loaded {i + 1}/{len(rows)} cases", flush=True)

    # --- 2. Random but reproducible train / validation split -------------------------------
    train_idx, val_idx = split(len(cases), cfg)
    train, val = [cases[i] for i in train_idx], [cases[i] for i in val_idx]

    # --- 3. Normalisation statistics, from the TRAINING cases only -------------------------
    # Using validation data here would leak information about the held-out cases into training.
    #
    # Fields: per-channel mean and std over all selected training points (blade + exit plane,
    # so the normalisation matches the points the model actually predicts). Average each case's
    # mean and mean square: every case has the same N, so this equals the overall mean.
    # double() avoids round-off when summing over many values.
    y_mean = torch.stack([c["y"].double().mean(0) for c in train]).mean(0)
    y_sq = torch.stack([c["y"].double().square().mean(0) for c in train]).mean(0)
    # Coordinates: subtract a per-axis mean but divide by a SINGLE scale (the largest axis std).
    # Scaling each axis separately would stretch the geometry anisotropically, so a ball-query
    # radius of 0.1 would cover different physical distances in x, r and rt.
    x_mean = torch.stack([c["x"].double().mean(0) for c in train]).mean(0)
    x_scale = torch.stack([c["x"].double().std(0) for c in train]).mean(0).max()
    # Design parameters and QoIs are one vector per case: plain mean/std over the training cases.
    # Standardising the QoIs matters a lot: p0 loss only spans ~1.44-1.98 %, so in raw units its
    # errors would be negligible next to the field loss.
    p, q = torch.stack([c["p"] for c in train]), torch.stack([c["q"] for c in train])
    constant = [k for k, s in zip(cfg["params"], p.std(0)) if s == 0]
    if constant:
        raise ValueError(f"inputs {constant} are constant in this dataset; remove them from cfg['params']")
    stats = {
        "x": (x_mean.float(), x_scale.float()),
        "y": (y_mean.float(), (y_sq - y_mean**2).sqrt().float()),  # std = sqrt(E[y^2] - E[y]^2)
        "p": (p.mean(0), p.std(0)),
        "q": (q.mean(0), q.std(0)),
    }

    # --- 4. Apply the same transform to every case (train and val) -------------------------
    # After this all model inputs and targets are roughly zero mean and unit variance.
    for c in cases:
        normalise(c, stats)
    # stats is returned so predictions can be converted back to physical units later
    return train, val, stats


def make_batch(cases, n_query, device):
    """Stack every point of each case, or a random subset of n_query of them."""
    # With n_query=None every point is used (slice(None) selects all rows). Otherwise a fresh
    # random subset is drawn each time, so over many epochs the model sees every point. The
    # same indices are used for x and y, so each target stays matched to its coordinate.
    idx = [torch.randperm(len(c["x"]))[:n_query] if n_query else slice(None) for c in cases]
    x = torch.stack([c["x"][i] for c, i in zip(cases, idx)])  # (B, N, 3)
    y = torch.stack([c["y"][i] for c, i in zip(cases, idx)])  # (B, N, 5)
    # Per-case quantities need no sampling: geom (B, n_geom, 3), p (B, 3), q (B, n_qoi)
    geom, p, q = (torch.stack([c[k] for c in cases]) for k in ("geom", "p", "q"))
    # non_blocking lets the CPU->GPU copy overlap with other work where possible
    return [t.to(device, non_blocking=True) for t in (x, geom, p, y, q)]


def forward(model, x, geom, p):
    """Give every point the case parameters as its features, and use them as global context too."""
    # GeoTransolver.forward(x, features, geometry, params):
    #   x        (B, N, 3)  where to predict
    #   features (B, N, F)  per-point inputs. Here the 3 design parameters, copied to every point
    #                       (expand creates a view, so this costs no memory). This lets every
    #                       point token see the design directly from the first layer.
    #   geometry (B, M, 3)  wall point cloud, read through the ball-query encoders and cross-attention
    #   params   (B, P)     global context, also read through cross-attention in every GALE block
    # Returns (fields (B, N, 5), qois (B, n_qoi)), both in normalised units.
    return model(x, p[:, None, :].expand(-1, x.shape[1], -1), geom, p)


def build_model(cfg):
    """The GeoTransolver used for training (and rebuilt from a checkpoint for plotting)."""
    m = cfg["model"]
    return GeoTransolver(
        feature_dim=len(cfg["params"]),   # per-point features = the design inputs (see forward)
        param_dim=len(cfg["params"]),     # global context = the same design inputs
        output_dim=len(CONSERVED),        # 5 conserved variables per point
        global_dim=len(cfg["qoi"]),       # size of the pooled QoI head
        hidden=m["hidden"], heads=m["heads"], slices=m["slices"], layers=m["layers"],
        geo_hidden=m["geo_hidden"], radii=tuple(m["radii"]), k=m["k"],
    )


@torch.no_grad()  # no gradients needed, which saves memory and time
def predict(model, c, cfg, device):
    """Predict every point of one normalised case: returns (fields (N, 5), qois (n_qoi,)), normalised."""
    n = len(c["x"])
    # [None] adds a batch dimension of 1
    geom, p = c["geom"][None].to(device), c["p"][None].to(device)
    y_hat = torch.empty(n, len(CONSERVED), device=device)
    q_hats = []
    # The physics attention pools over all points it is given, so prediction should see
    # point sets like those in training. With n_query=None training uses every point, so
    # predict them all at once. If training subsamples, shuffle the points into ~equal
    # chunks of about n_query, predict each, and write results back to their positions.
    n_chunks = max(1, n // cfg["n_query"]) if cfg["n_query"] else 1
    for chunk in torch.randperm(n).tensor_split(n_chunks):
        yc, qc = forward(model, c["x"][chunk][None].to(device), geom, p)
        y_hat[chunk.to(device)] = yc[0]
        q_hats.append(qc[0])  # every chunk gives its own QoI estimate
    return y_hat, torch.stack(q_hats).mean(0)  # average the per-chunk QoI estimates


@torch.no_grad()
def validate(model, cases, stats, cfg, device):
    """Predict every point of each case and report errors on the blade and exit plane separately."""
    model.eval()
    y_mean, y_std = (t.to(device) for t in stats["y"])
    q_std = stats["q"][1].to(device)
    rel_l2 = {"blade": [], "exit": []}
    q_err = []
    for c in cases:
        y_hat, q_hat = predict(model, c, cfg, device)
        y = c["y"].to(device)
        # Relative L2 error per variable, in physical units:
        #   numerator:   (y_hat - y) * std  = physical error (the mean cancels in the difference)
        #   denominator: y * std + mean     = physical field
        # .norm(dim=0) takes the L2 norm over the region's points, leaving one value per variable.
        for region, rows in (("blade", slice(None, c["n_blade"])), ("exit", slice(c["n_blade"], None))):
            err = ((y_hat[rows] - y[rows]) * y_std).norm(dim=0)
            rel_l2[region].append(err / (y[rows] * y_std + y_mean).norm(dim=0))
        # QoI: absolute error in physical units
        q_err.append((q_hat - c["q"].to(device)).abs() * q_std)
    model.train()  # back to training mode for the next epoch
    # Average over validation cases, e.g. {"val_blade_ro": 0.0012, ..., "val_mae_pp_p0_loss": 0.000412}
    metrics = {}
    for region, errs in rel_l2.items():
        for v, e in zip(CONSERVED, torch.stack(errs).mean(0).tolist()):
            metrics[f"val_{region}_{v}"] = e
    for k, e in zip(cfg["qoi"], torch.stack(q_err).mean(0).tolist()):
        metrics[f"val_mae_{k}"] = e
    return metrics


def format_metrics(metrics, cfg):
    """Validation metrics as one printable line."""
    fields = " | ".join(
        f"{region} " + " ".join(f"{v}:{metrics[f'val_{region}_{v}']:.3%}" for v in CONSERVED)
        for region in ("blade", "exit")
    )
    qois = " ".join(f"{k}:{metrics[f'val_mae_{k}']:.3g}" for k in cfg["qoi"])
    return f"| val relL2 {fields} | val MAE {qois}"


def train(cfg=CFG, data=None, results=RESULTS, on_checkpoint=None):
    """Train one run, saving everything in a new numbered folder results/<run_prefix>_NN:
        config.json        the full config plus run details (written first, so even a crashed run has it)
        history.csv        one row per epoch: training losses, learning rate and validation metrics
        geotransolver.pt   weights + normalisation stats + config, rewritten at every validation
        figures/           filled in later by the plotting scripts
    data defaults to the local folder of cfg["dataset"]; on_checkpoint is called after each save
    (Modal uses it to commit the volume).
    """
    cfg = resolve(cfg)
    data = data or data_dir(cfg)
    run = new_run_dir(results, cfg["run_prefix"])
    torch.manual_seed(cfg["seed"])  # reproducible initialisation and batch order
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Allow TF32 matrix multiplies on recent NVIDIA GPUs: much faster, slightly less precise
    torch.set_float32_matmul_precision("high")
    train_cases, val_cases, stats = prepare(cfg, data)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    info = {
        "run": run.name, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "data": str(data),
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu", "torch": torch.__version__,
        "parameters": n_params, "n_train": len(train_cases), "n_val": len(val_cases),
        "val_cases": [c["case"] for c in val_cases],
    }
    (run / "config.json").write_text(json.dumps({"cfg": cfg, "run": info}, indent=2))
    print(f"run folder: {run}", flush=True)
    # Zero-initialised outputs: the untrained model predicts the training mean
    # (targets are standardised, so 0 = training mean). Only the last layer is zeroed; earlier
    # layers keep random weights, so units still differ and receive different gradients.
    with torch.no_grad():
        for head in (model.output_head, model.global_head):
            head[-1].weight.zero_()
            head[-1].bias.zero_()

    # AdamW = Adam with weight decay applied directly to the weights, not added to the gradient
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # Learning rate follows half a cosine from lr down to min_lr over the whole run (stepped per epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["min_lr"])
    print(f"{len(train_cases)} train / {len(val_cases)} val cases, {n_params:,} parameters on {device}")

    def save():
        # Weights together with the normalisation stats and config: the stats are needed to
        # normalise new inputs and convert predictions back to physical units at inference time.
        torch.save({"model": model.state_dict(), "stats": stats, "cfg": cfg, "run": run.name},
                   run / "geotransolver.pt")
        if on_checkpoint:
            on_checkpoint()

    # history.csv gets the same columns every epoch; validation columns are blank between validations
    history_cols = ["epoch", "field_loss", "qoi_loss", "lr", "seconds"]
    history_cols += [f"val_{r}_{v}" for r in ("blade", "exit") for v in CONSERVED]
    history_cols += [f"val_mae_{k}" for k in cfg["qoi"]]
    with open(run / "history.csv", "w", newline="") as fh:
        csv.writer(fh).writerow(history_cols)

    for epoch in range(1, cfg["epochs"] + 1):
        started = time.perf_counter()
        # Shuffle the case order every epoch so batches are mixed differently each time
        order = torch.randperm(len(train_cases)).tolist()
        totals = torch.zeros(2)  # running sums of [field_loss, qoi_loss], weighted by cases
        for b in range(0, len(order), cfg["batch_size"]):
            batch = [train_cases[i] for i in order[b:b + cfg["batch_size"]]]
            x, geom, p, y, q = make_batch(batch, cfg["n_query"], device)
            y_hat, q_hat = forward(model, x, geom, p)
            # Both losses are MSE in normalised units, so every variable and QoI counts roughly
            # equally regardless of its physical magnitude (e.g. roe ~1e5 vs ro ~1).
            field_loss = (y_hat - y).square().mean()
            qoi_loss = (q_hat - q).square().mean()
            # The QoI term is up-weighted: it is only n_qoi numbers per case against ~12.3k x 5
            # field values, and these are the quantities we most care about.
            loss = field_loss + cfg["w_qoi"] * qoi_loss

            optimizer.zero_grad(set_to_none=True)  # clear gradients from the previous step
            loss.backward()                        # backpropagate: fill .grad for every parameter
            # If the total gradient norm is above `clip`, scale every gradient down to that norm.
            # This protects against occasional large gradients destabilising training.
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            optimizer.step()                       # update the weights
            # .item() copies to the CPU; multiply by batch size so the last, smaller batch
            # is weighted correctly in the epoch average
            totals += torch.tensor([field_loss.item(), qoi_loss.item()]) * len(batch)
        scheduler.step()  # lower the learning rate once per epoch

        # Average training losses over the epoch (still in normalised units)
        field, qoi = (totals / len(train_cases)).tolist()
        lr = scheduler.get_last_lr()[0]
        msg = f"epoch {epoch:4d}  field {field:.4e}  qoi {qoi:.4e}  lr {lr:.1e}"
        row = {"epoch": epoch, "field_loss": field, "qoi_loss": qoi, "lr": lr}
        # Validation (and a checkpoint) is run periodically and at the end, not every epoch
        if epoch % cfg["val_every"] == 0 or epoch == cfg["epochs"]:
            metrics = validate(model, val_cases, stats, cfg, device)
            row.update(metrics)
            msg += "  " + format_metrics(metrics, cfg)
            save()
        row["seconds"] = time.perf_counter() - started
        with open(run / "history.csv", "a", newline="") as fh:
            csv.writer(fh).writerow([row.get(k, "") for k in history_cols])
        print(msg, flush=True)  # flush so lines appear straight away in Modal's logs
    print(f"saved {run / 'geotransolver.pt'}", flush=True)


if __name__ == "__main__":
    train()
