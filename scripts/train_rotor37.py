"""Train GeoTransolver on the PLAID Rotor37 dataset (NASA Rotor 37 compressor blade, Safran),
using the same recipe as train.py: training-only normalisation, a held-out validation split,
zero-initialised outputs, an explicit QoI loss, and AdamW with a cosine schedule and gradient clipping.

The dataset
-----------
datasets/Rotor37 is the HuggingFace copy saved by datasets/download_dataset.py: 1200 samples, each
a pickled PLAID sample (a CGNS tree + scalars). Every sample is the blade surface of one perturbed
Rotor 37 geometry, meshed with the SAME 29,773-node quad mesh (only the node positions change):
    inputs   Omega, P (rotation speed, outlet pressure) + the surface normals at every node
    outputs  Density, Pressure, Temperature at every node
             Massflow, Compression_ratio, Efficiency (one value per sample, the QoIs)
Samples 0-999 are the training pool (the official train_8 ... train_1000 subsets are nested in it);
1000-1199 are the official test split, which is published WITHOUT outputs, so we hold out our own
validation cases from the training pool instead.

Overview of the pipeline
------------------------
1. load_rotor37(): decode every sample once and cache the arrays in rotor37_cache.npz.
2. prepare():      split into train / validation and standardise using training statistics only.
3. make_batch():   each optimisation step picks a few cases and uses every surface node
                   (optionally a random subset of n_query of them).
4. forward():      per case the model sees the nodes (+ normals and Omega, P as features), a wall
                   point cloud of the blade, and (Omega, P); it predicts the 3 fields and 3 QoIs.
5. train():        AdamW + cosine learning-rate decay, loss = field MSE + w_qoi * QoI MSE.
6. validate():     relative L2 errors of the fields and absolute / relative errors of the QoIs.

Run from the repo root:  python scripts/train_rotor37.py
"""
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Make the repo root importable (for `models/`) regardless of where the script is run from
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models.geotransolver import GeoTransolver
from train import new_run_dir

DATA = ROOT / "datasets" / "Rotor37"  # on Modal: /data/Rotor37 (only rotor37_cache.npz is needed there)
CACHE = "rotor37_cache.npz"            # decoded arrays, written next to the Arrow shards on first run
FIELDS = ("Density", "Pressure", "Temperature")      # per-node outputs
NORMALS = ("NormalsX", "NormalsY", "NormalsZ")       # per-node inputs
INPUTS = ("Omega", "P")                              # per-sample scalar inputs
QOIS = ("Massflow", "Compression_ratio", "Efficiency")  # per-sample scalar outputs
N_POOL = 1000  # samples 0..999 have outputs; 1000..1199 (the official test split) do not
RESULTS = ROOT / "results"

CFG = dict(
    run_prefix="rotor37",  # runs are saved as results/<run_prefix>_01, _02, ...
    train_split="train_1000",  # official training subset: train_8/16/32/64/125/250/500/1000
    seed=42,        # fixes the validation cases, model initialisation and data order
    n_val=100,      # cases held out of train_1000 for validation (the same ones for every train_split)
    epochs=300,     # one epoch = one pass over every training case
    batch_size=4,   # cases per optimisation step
    lr=1e-3,        # peak learning rate (start of the cosine schedule)
    min_lr=1e-5,    # learning rate the cosine schedule decays to by the last epoch
    weight_decay=1e-4,  # AdamW's decoupled L2 penalty on the weights
    clip=1.0,       # maximum gradient norm; larger gradients are rescaled down to this
    n_query=None,   # nodes sampled per case per step; None = all 29,773
    n_geom=2048,    # blade nodes in the geometry context
    w_qoi=4.0,      # weight of the QoI loss relative to the field loss
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


def child(node, name):
    """The child called name of a CGNS node, stored as [name, value, children, type]."""
    return next(c for c in node[2] if c[0] == name)


def decode_sample(blob):
    """One pickled PLAID sample -> dict of numpy arrays (fields and QoIs are None for test samples)."""
    s = pickle.loads(blob)
    tree = next(iter(s["meshes"].values()))  # a single time step
    zone = child(child(tree, "Base_2_3"), "Zone")
    coords = child(zone, "GridCoordinates")
    point_data = {c[0]: c[1] for c in child(zone, "PointData")[2]}
    has_outputs = all(k in point_data for k in FIELDS)
    return {
        "coords": np.stack([child(coords, f"Coordinate{a}")[1] for a in "XYZ"], 1),  # (N, 3)
        "normals": np.stack([point_data[k] for k in NORMALS], 1),                   # (N, 3)
        "fields": np.stack([point_data[k] for k in FIELDS], 1) if has_outputs else None,
        "inputs": np.array([s["scalars"][k] for k in INPUTS]),
        "qois": np.array([s["scalars"][k] for k in QOIS]) if has_outputs else None,
    }


def load_rotor37(data):
    """Arrays for the 1000 samples that have outputs, decoded once and cached in data/CACHE:
        coords (S, N, 3), normals (S, N, 3), fields (S, N, 3), inputs (S, 2), qois (S, 3), float32,
    plus "splits", the official train_* subsets (sample ids) from dataset_info.json."""
    data = Path(data)
    if not (data / CACHE).exists():
        import pyarrow as pa  # only needed to build the cache, so not required on Modal

        shards = sorted((data / "all_samples").glob("data-*.arrow"))
        if not shards:
            raise FileNotFoundError(f"no Arrow shards in {data / 'all_samples'}; run datasets/download_dataset.py")
        samples = []
        # Row order across the shards is the sample id (0..1199); stop once the training pool is read
        for shard in shards:
            with pa.memory_map(str(shard)) as f:
                for blob in pa.ipc.open_stream(f).read_all().column("sample"):
                    if len(samples) == N_POOL:
                        break
                    samples.append(decode_sample(blob.as_py()))
                    if len(samples) % 100 == 0:
                        print(f"decoded {len(samples)}/{N_POOL} samples", flush=True)
        missing = [i for i, s in enumerate(samples) if s["fields"] is None]
        if len(samples) < N_POOL or missing:
            raise ValueError(f"expected outputs for samples 0..{N_POOL - 1}; missing for {missing[:10]}")
        info = json.loads((data / "all_samples" / "dataset_info.json").read_text())
        splits = {k: np.array(v) for k, v in info["description"]["split"].items() if k.startswith("train_")}
        np.savez(data / CACHE, **{k: np.stack([s[k] for s in samples]).astype(np.float32) for k in samples[0]},
                 **{f"split_{k}": v for k, v in splits.items()})
        print(f"cached {data / CACHE}", flush=True)
    with np.load(data / CACHE) as z:
        out = {k: torch.from_numpy(z[k]) for k in ("coords", "normals", "fields", "inputs", "qois")}
        out["splits"] = {k[len("split_"):]: z[k].tolist() for k in z.files if k.startswith("split_")}
    return out


def split(d, cfg):
    """(train, val) sample ids. The validation cases are drawn from train_1000 with a dedicated
    Generator, so they depend only on cfg["seed"] and are the same whichever train_split is used;
    training uses cfg["train_split"] minus any validation cases."""
    perm = torch.randperm(N_POOL, generator=torch.Generator().manual_seed(cfg["seed"])).tolist()
    val = sorted(perm[:cfg["n_val"]])
    if cfg["train_split"] not in d["splits"]:
        raise ValueError(f"unknown train_split {cfg['train_split']!r}; choose from {sorted(d['splits'])}")
    train = [i for i in d["splits"][cfg["train_split"]] if i not in set(val)]
    return train, val


def prepare(cfg, data):
    """Load every case, split, and normalise with training-set statistics only."""
    d = load_rotor37(data)
    train_idx, val_idx = split(d, cfg)
    n_nodes = d["coords"].shape[1]
    # The mesh topology is shared, so the same evenly strided node ids give a consistent wall cloud
    geom_idx = torch.linspace(0, n_nodes - 1, cfg["n_geom"]).long()

    # --- Normalisation statistics, from the TRAINING cases only -----------------------------
    t = torch.tensor(train_idx)
    coords, fields = d["coords"][t].double(), d["fields"][t].double()
    stats = {
        # Coordinates: per-axis mean but ONE scale (the largest axis std), so the geometry is not
        # stretched anisotropically and a ball-query radius means the same distance along every axis
        "x": (coords.mean((0, 1)).float(), coords.std(1).mean(0).max().float()),
        "y": (fields.mean((0, 1)).float(), fields.std((0, 1)).float()),  # per-field mean / std
        "p": (d["inputs"][t].mean(0), d["inputs"][t].std(0)),
        "q": (d["qois"][t].mean(0), d["qois"][t].std(0)),
    }

    def case(i):
        x = (d["coords"][i] - stats["x"][0]) / stats["x"][1]
        p = (d["inputs"][i] - stats["p"][0]) / stats["p"][1]
        return {
            "case": i,
            "x": x,                                        # (N, 3) node positions
            "n": d["normals"][i],                          # (N, 3) unit normals, used as is
            "geom": x[geom_idx],                           # (n_geom, 3) wall point cloud
            "y": (d["fields"][i] - stats["y"][0]) / stats["y"][1],  # (N, 3) fields
            "p": p,                                        # (2,) Omega, P
            "q": (d["qois"][i] - stats["q"][0]) / stats["q"][1],    # (3,) QoIs
        }

    return [case(i) for i in train_idx], [case(i) for i in val_idx], stats


def make_batch(cases, n_query, device):
    """Stack every node of each case, or a fresh random subset of n_query of them."""
    idx = [torch.randperm(len(c["x"]))[:n_query] if n_query else slice(None) for c in cases]
    x, n, y = (torch.stack([c[k][i] for c, i in zip(cases, idx)]) for k in ("x", "n", "y"))
    geom, p, q = (torch.stack([c[k] for c in cases]) for k in ("geom", "p", "q"))
    return [t.to(device, non_blocking=True) for t in (x, n, geom, p, y, q)]


def forward(model, x, n, geom, p):
    """Per-node features = the normals plus (Omega, P) copied to every node; (Omega, P) is also
    the global context. Returns (fields (B, N, 3), qois (B, 3)) in normalised units."""
    features = torch.cat([n, p[:, None, :].expand(-1, x.shape[1], -1)], dim=-1)
    return model(x, features, geom, p)


def build_model(cfg):
    m = cfg["model"]
    return GeoTransolver(
        feature_dim=len(NORMALS) + len(INPUTS), param_dim=len(INPUTS),
        output_dim=len(FIELDS), global_dim=len(QOIS),
        hidden=m["hidden"], heads=m["heads"], slices=m["slices"], layers=m["layers"],
        geo_hidden=m["geo_hidden"], radii=tuple(m["radii"]), k=m["k"],
    )


@torch.no_grad()
def predict(model, c, cfg, device):
    """Predict every node of one normalised case: returns (fields (N, 3), qois (3,)), normalised.
    If training subsamples n_query nodes, predict in shuffled chunks of about that size so the
    physics attention pools over point sets like those seen in training."""
    n = len(c["x"])
    geom, p = c["geom"][None].to(device), c["p"][None].to(device)
    y_hat = torch.empty(n, len(FIELDS), device=device)
    q_hats = []
    n_chunks = max(1, n // cfg["n_query"]) if cfg["n_query"] else 1
    for chunk in torch.randperm(n).tensor_split(n_chunks):
        yc, qc = forward(model, c["x"][chunk][None].to(device), c["n"][chunk][None].to(device), geom, p)
        y_hat[chunk.to(device)] = yc[0]
        q_hats.append(qc[0])
    return y_hat, torch.stack(q_hats).mean(0)


@torch.no_grad()
def validate(model, cases, stats, cfg, device):
    """Relative L2 error of each field and absolute / relative error of each QoI, in physical units,
    averaged over the validation cases."""
    model.eval()
    y_mean, y_std = (t.to(device) for t in stats["y"])
    q_mean, q_std = (t.to(device) for t in stats["q"])
    rel_l2, q_abs, q_rel = [], [], []
    for c in cases:
        y_hat, q_hat = predict(model, c, cfg, device)
        y, q = c["y"].to(device), c["q"].to(device)
        rel_l2.append(((y_hat - y) * y_std).norm(dim=0) / (y * y_std + y_mean).norm(dim=0))
        q_abs.append((q_hat - q).abs() * q_std)
        q_rel.append(q_abs[-1] / (q * q_std + q_mean).abs())
    model.train()
    metrics = {f"val_{k}": e for k, e in zip(FIELDS, torch.stack(rel_l2).mean(0).tolist())}
    metrics |= {f"val_mae_{k}": e for k, e in zip(QOIS, torch.stack(q_abs).mean(0).tolist())}
    metrics |= {f"val_mre_{k}": e for k, e in zip(QOIS, torch.stack(q_rel).mean(0).tolist())}
    return metrics


def format_metrics(metrics):
    fields = " ".join(f"{k}:{metrics[f'val_{k}']:.3%}" for k in FIELDS)
    qois = " ".join(f"{k}:{metrics[f'val_mre_{k}']:.3%}" for k in QOIS)
    return f"| val relL2 {fields} | val rel err {qois}"


def train(cfg=CFG, data=DATA, results=RESULTS, on_checkpoint=None):
    """Train one run, saving everything in a new numbered folder results/<run_prefix>_NN:
        config.json        the full config plus run details (written first, so even a crashed run has it)
        history.csv        one row per epoch: training losses, learning rate and validation metrics
        geotransolver.pt   weights + normalisation stats + config, rewritten at every validation
    on_checkpoint is called after each save (Modal uses it to commit the volume).
    """
    run = new_run_dir(results, cfg["run_prefix"])
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")  # TF32 on recent NVIDIA GPUs
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
    with torch.no_grad():
        for head in (model.output_head, model.global_head):
            head[-1].weight.zero_()
            head[-1].bias.zero_()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["min_lr"])
    print(f"{len(train_cases)} train / {len(val_cases)} val cases, {n_params:,} parameters on {device}")

    def save():
        torch.save({"model": model.state_dict(), "stats": stats, "cfg": cfg, "run": run.name},
                   run / "geotransolver.pt")
        if on_checkpoint:
            on_checkpoint()

    history_cols = ["epoch", "field_loss", "qoi_loss", "lr", "seconds"]
    history_cols += [f"val_{k}" for k in FIELDS]
    history_cols += [f"val_{e}_{k}" for e in ("mae", "mre") for k in QOIS]
    with open(run / "history.csv", "w", newline="") as fh:
        csv.writer(fh).writerow(history_cols)

    for epoch in range(1, cfg["epochs"] + 1):
        started = time.perf_counter()
        order = torch.randperm(len(train_cases)).tolist()
        totals = torch.zeros(2)  # running sums of [field_loss, qoi_loss], weighted by cases
        for b in range(0, len(order), cfg["batch_size"]):
            batch = [train_cases[i] for i in order[b:b + cfg["batch_size"]]]
            x, n, geom, p, y, q = make_batch(batch, cfg["n_query"], device)
            y_hat, q_hat = forward(model, x, n, geom, p)
            field_loss = (y_hat - y).square().mean()
            qoi_loss = (q_hat - q).square().mean()
            loss = field_loss + cfg["w_qoi"] * qoi_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            optimizer.step()
            totals += torch.tensor([field_loss.item(), qoi_loss.item()]) * len(batch)
        scheduler.step()

        field, qoi = (totals / len(train_cases)).tolist()
        lr = scheduler.get_last_lr()[0]
        msg = f"epoch {epoch:4d}  field {field:.4e}  qoi {qoi:.4e}  lr {lr:.1e}"
        row = {"epoch": epoch, "field_loss": field, "qoi_loss": qoi, "lr": lr}
        if epoch % cfg["val_every"] == 0 or epoch == cfg["epochs"]:
            metrics = validate(model, val_cases, stats, cfg, device)
            row.update(metrics)
            msg += "  " + format_metrics(metrics)
            save()
        row["seconds"] = time.perf_counter() - started
        with open(run / "history.csv", "a", newline="") as fh:
            csv.writer(fh).writerow([row.get(k, "") for k in history_cols])
        print(msg, flush=True)
    print(f"saved {run / 'geotransolver.pt'}", flush=True)


if __name__ == "__main__":
    # Optional JSON overrides of CFG, e.g. python scripts/train_rotor37.py '{"train_split": "train_125"}'
    overrides = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    cfg = {**CFG, **overrides, "model": {**CFG["model"], **overrides.get("model", {})}}
    train(cfg)
