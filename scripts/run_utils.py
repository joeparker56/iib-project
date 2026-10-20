"""Find and load a training run folder (results/<run_prefix>_NN) for the plotting scripts."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402


def resolve_run(run=None):
    """The run folder to use: the given path or name, else the most recently modified run in results/."""
    if run:
        path = Path(run)
        return path if path.exists() else T.RESULTS / run
    runs = [d for d in T.RESULTS.glob("*_[0-9][0-9]*") if (d / "geotransolver.pt").exists()]
    if not runs:
        sys.exit(f"no trained runs in {T.RESULTS}; download one from Modal or pass --run")
    return max(runs, key=lambda d: (d / "geotransolver.pt").stat().st_mtime)


def load_run(run_dir, device):
    """Rebuild the model from a run's checkpoint. Returns (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(Path(run_dir) / "geotransolver.pt", map_location="cpu", weights_only=False)
    # Checkpoints from before run folders lack these entries. They were all trained on grid125_R23
    # with the then hard-coded model settings, which match the defaults in T.CFG["model"].
    ckpt["cfg"] = T.resolve({"dataset": "grid125_R23", "params": None, "model": T.CFG["model"], **ckpt["cfg"]})
    ckpt.setdefault("run", Path(run_dir).name)
    model = T.build_model(ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def add_run_arg(ap):
    ap.add_argument("--run", default=None,
                    help="run folder or name, e.g. geotransolver_surface_01 (default: latest run in results/)")
