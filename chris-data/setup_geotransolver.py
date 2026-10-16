"""Minimal GeoTransolver setup (physicsnemo.models.geotransolver). Fill in the blanks (None)."""
import torch
from physicsnemo.models.geotransolver import GeoTransolver

# ---- Fill these in -------------------------------------------------------
FUNCTIONAL_DIM = None   # int: channels of the local input at each point (C)
OUT_DIM = None          # int: channels predicted at each point
GEOMETRY_DIM = None     # int | None: per-point geometry features (e.g. 3 for x, r, rt)
GLOBAL_DIM = None       # int | None: global features per case (e.g. 3 for sweep, lean, theta)
STRUCTURED_SHAPE = None # (I, J, K) to use the structured 3D path, or None for flat (B, N, C) tokens
# --------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GeoTransolver(
    functional_dim=FUNCTIONAL_DIM,
    out_dim=OUT_DIM,
    geometry_dim=GEOMETRY_DIM,
    global_dim=GLOBAL_DIM,
    structured_shape=STRUCTURED_SHAPE,
    n_layers=4,
    n_hidden=256,
    n_head=8,
    slice_num=32,
    use_te=False,
).to(device)

print(f"{sum(p.numel() for p in model.parameters()):,} parameters")

# ---- Forward pass (fill in the tensors) ----------------------------------
# local_embedding : (B, N, FUNCTIONAL_DIM)   or (B, I, J, K, FUNCTIONAL_DIM) if structured
# geometry        : (B, N, GEOMETRY_DIM)     optional
# global_embedding: (B, N_g, GLOBAL_DIM)     optional
local_embedding = None
geometry = None
global_embedding = None

if local_embedding is not None:
    out = model(
        local_embedding.to(device),
        geometry=None if geometry is None else geometry.to(device),
        global_embedding=None if global_embedding is None else global_embedding.to(device),
    )
    print(out.shape)  # (B, N, OUT_DIM), or spatial layout if structured
