"""Run train_geotransolver.py (or train_geotransolver_surface.py) on a Modal GPU.

One-off: upload the dataset to a Modal volume (from the chris-data folder):
    modal volume create geotransolver-data
    modal volume put geotransolver-data grid125_R23 /grid125_R23

Train (extra args are passed straight to the training script):
    modal run modal_train.py --args "--epochs 100 --batch_size 4"
    modal run --detach modal_train.py --args "--epochs 300"     # keeps running if you disconnect
Blade-surface + plane model (train_geotransolver_surface.py), saved under its own run name:
    modal run --detach modal_train.py --script train_geotransolver_surface.py --run-name surface --args "--epochs 300"

Checkpoints land in the `geotransolver-runs` volume under /<run_name>/best.pt:
    modal volume get geotransolver-runs /geotransolver/best.pt .
"""
import shlex
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CODE = "/root/code"

app = modal.App("geotransolver-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
    .add_local_file(HERE / "train_geotransolver.py", f"{CODE}/train_geotransolver.py")
    .add_local_file(HERE / "train_geotransolver_surface.py", f"{CODE}/train_geotransolver_surface.py")
    .add_local_file(HERE / "geotransolver_torch.py", f"{CODE}/geotransolver_torch.py")
    .add_local_file(HERE / "tensor_dataset.py", f"{CODE}/tensor_dataset.py")
)

data_vol = modal.Volume.from_name("geotransolver-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("geotransolver-runs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=65536,  # all cases are loaded into RAM up front
    timeout=24 * 60 * 60,
    volumes={"/data": data_vol, "/runs": runs_vol},
)
def train(args: str = "", run_name: str = "geotransolver", script: str = "train_geotransolver.py"):
    cmd = [
        "python", "-u", f"{CODE}/{script}",
        "--data", "/data/grid125_R23",
        "--out", f"/runs/{run_name}",
        *shlex.split(args),
    ]
    print("running:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True, cwd=CODE)
    finally:
        runs_vol.commit()  # persist checkpoints even if training fails/is interrupted


@app.local_entrypoint()
def main(args: str = "", run_name: str = "geotransolver", script: str = "train_geotransolver.py"):
    train.remote(args, run_name, script)
