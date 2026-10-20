"""Run scripts/train_rotor37.py on a Modal GPU.

The decoded Rotor37 arrays live in the `geotransolver-data` volume as /Rotor37/rotor37_cache.npz.
Build the cache locally first (any local run of train_rotor37.py writes it), then upload it once,
from the repo root (PowerShell):
    modal volume put geotransolver-data datasets/Rotor37/rotor37_cache.npz /Rotor37/rotor37_cache.npz

Train (--cfg overrides entries of CFG in train_rotor37.py, as JSON; "model" entries are merged key by key):
    modal run scripts/modal_train_rotor37.py --cfg '{"epochs": 2, "val_every": 1}'
    modal run --detach scripts/modal_train_rotor37.py
    modal run --detach scripts/modal_train_rotor37.py --cfg '{"train_split": "train_125", "epochs": 500}'
    modal run --detach scripts/modal_train_rotor37.py --cfg '{"n_query": 8192, "model": {"hidden": 256}}'

Each run is saved in the `geotransolver-runs` volume as /<run_prefix>_NN (numbered automatically,
printed at the start of the log). Download the whole folder into results/ (run in PowerShell):
    modal volume get geotransolver-runs /rotor37_01 results/
"""
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
PROJECT = "/root/project"  # same layout as the repo, so train_rotor37.py's imports work unchanged

app = modal.App("geotransolver-train-rotor37")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
    .add_local_dir(ROOT / "models", f"{PROJECT}/models", ignore=["__pycache__", "*.pt"])
    .add_local_file(ROOT / "scripts" / "train.py", f"{PROJECT}/scripts/train.py")  # for new_run_dir
    .add_local_file(ROOT / "scripts" / "train_rotor37.py", f"{PROJECT}/scripts/train_rotor37.py")
)

data_vol = modal.Volume.from_name("geotransolver-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("geotransolver-runs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=32768,  # all 1000 surfaces (~1 GB as float32) are held in RAM
    timeout=24 * 60 * 60,
    volumes={"/data": data_vol, "/runs": runs_vol},
)
def train(cfg_overrides: dict):
    sys.path.insert(0, f"{PROJECT}/scripts")
    import train_rotor37 as train_script

    cfg = {**train_script.CFG, **cfg_overrides}
    cfg["model"] = {**train_script.CFG["model"], **cfg_overrides.get("model", {})}
    print("cfg:", cfg, flush=True)
    if not Path("/data/Rotor37", train_script.CACHE).exists():
        raise FileNotFoundError("upload the cache first: modal volume put geotransolver-data "
                                "datasets/Rotor37/rotor37_cache.npz /Rotor37/rotor37_cache.npz")
    try:
        # commit after every checkpoint, so a detached run that dies still leaves its latest weights
        train_script.train(cfg, data=Path("/data/Rotor37"), results=Path("/runs"),
                           on_checkpoint=runs_vol.commit)
    finally:
        runs_vol.commit()  # persist whatever was written even if training fails


@app.local_entrypoint()
def main(cfg: str = "{}"):
    train.remote(json.loads(cfg))
