"""Run scripts/train.py on a Modal GPU.

Datasets live in the `geotransolver-data` volume as /<dataset name> (a key of DATASETS in train.py).
Upload each one once, from the repo root (PowerShell):
    modal volume put geotransolver-data datasets/chris-surrogate-lean-sweep/grid125_R23 /grid125_R23
    modal volume put geotransolver-data datasets/chris-cfd-outputs/conv1000_c4v2 /conv1000_c4v2

Train (--cfg overrides entries of CFG in train.py, as JSON; "model" entries are merged key by key):
    modal run scripts/modal_train.py --cfg '{"epochs": 2, "val_every": 1}'
    modal run --detach scripts/modal_train.py
    modal run --detach scripts/modal_train.py --cfg '{"w_qoi": 8.0, "model": {"hidden": 256}}'
    modal run --detach scripts/modal_train.py --cfg '{"dataset": "conv1000_c4v2", "n_val": 100}'

Each run is saved in the `geotransolver-runs` volume as /<run_prefix>_NN (numbered automatically,
printed at the start of the log). Download the whole folder into results/ (run in PowerShell):
    modal volume get geotransolver-runs /geotransolver_surface_01 results/
"""
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
PROJECT = "/root/project"  # same layout as the repo, so train.py's imports work unchanged

app = modal.App("geotransolver-train-v2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
    .add_local_dir(ROOT / "models", f"{PROJECT}/models", ignore=["__pycache__", "*.pt"])
    .add_local_file(ROOT / "scripts" / "train.py", f"{PROJECT}/scripts/train.py")
)

data_vol = modal.Volume.from_name("geotransolver-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("geotransolver-runs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=32768,  # every case's blade + exit-plane points are held in RAM
    timeout=24 * 60 * 60,
    volumes={"/data": data_vol, "/runs": runs_vol},
)
def train(cfg_overrides: dict):
    sys.path.insert(0, f"{PROJECT}/scripts")
    import train as train_script

    cfg = {**train_script.CFG, **cfg_overrides}
    cfg["model"] = {**train_script.CFG["model"], **cfg_overrides.get("model", {})}
    print("cfg:", cfg, flush=True)
    try:
        # commit after every checkpoint, so a detached run that dies still leaves its latest weights
        train_script.train(cfg, data=Path("/data") / cfg["dataset"], results=Path("/runs"),
                           on_checkpoint=runs_vol.commit)
    finally:
        runs_vol.commit()  # persist whatever was written even if training fails


@app.local_entrypoint()
def main(cfg: str = "{}"):
    train.remote(json.loads(cfg))
