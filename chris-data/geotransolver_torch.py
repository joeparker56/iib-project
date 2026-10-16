"""GeoTransolver in plain PyTorch (unstructured / flat-token path, single input stream).

Mirrors physicsnemo.models.geotransolver.GeoTransolver:
  - Physics attention (Transolver): tokens are softly assigned to `slice_num` learned slices,
    attention runs between slice tokens, and the result is scattered back to the tokens.
  - GALE: slice tokens additionally cross-attend to a context built from geometry / global
    features, and the self- and cross-attention outputs are blended with a learned sigmoid gate.
  - Context: geometry (B, N, C_geo) and global (B, N_g, C_g) features are each tokenised into
    (B, heads, slices, dim_head) by a ContextProjector and concatenated along the last dim.
Not implemented: structured Conv3d path, local ball-query features, Transolver++, time input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def slice_tokens(x_mid, fx_mid, in_project_slice, temperature):
    """x_mid, fx_mid: (B, N, H, D) -> slice weights (B, N, H, S), slice tokens (B, H, S, D)."""
    logits = in_project_slice(x_mid)
    w = F.softmax(logits / temperature.clamp(0.5, 5), dim=-1)
    norm = w.sum(dim=1) + 1e-2  # (B, H, S): total weight per slice over all tokens
    tok = torch.einsum("bnhs,bnhd->bhsd", w / norm[:, None], fx_mid)
    return w, tok


def make_slice_projection(dim_head, slice_num):
    lin = nn.Linear(dim_head, slice_num)
    nn.init.orthogonal_(lin.weight)  # diverse slices at init
    return lin


class ContextProjector(nn.Module):
    """(B, N, C) -> (B, H, S, D): the 'encode' half of physics attention, used for context."""

    def __init__(self, dim, heads, dim_head, slice_num):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        self.in_project_x = nn.Linear(dim, heads * dim_head)
        self.in_project_fx = nn.Linear(dim, heads * dim_head)
        self.in_project_slice = make_slice_projection(dim_head, slice_num)
        self.temperature = nn.Parameter(torch.full((1, 1, heads, 1), 0.5))

    def forward(self, x):
        B, N, _ = x.shape
        x_mid = self.in_project_x(x).view(B, N, self.heads, self.dim_head)
        fx_mid = self.in_project_fx(x).view(B, N, self.heads, self.dim_head)
        return slice_tokens(x_mid, fx_mid, self.in_project_slice, self.temperature)[1]


class GALE(nn.Module):
    """Physics attention with optional cross-attention to a context (B, H, S_c, D_c)."""

    def __init__(self, dim, heads, slice_num, context_dim=0, dropout=0.0):
        super().__init__()
        self.heads, self.dim_head = heads, dim // heads
        d = self.dim_head
        self.in_project_x = nn.Linear(dim, dim)
        self.in_project_fx = nn.Linear(dim, dim)
        self.in_project_slice = make_slice_projection(d, slice_num)
        self.temperature = nn.Parameter(torch.full((1, 1, heads, 1), 0.5))
        self.qkv_project = nn.Linear(d, 3 * d, bias=False)
        if context_dim > 0:
            self.cross_q = nn.Linear(d, d)
            self.cross_k = nn.Linear(context_dim, d)
            self.cross_v = nn.Linear(context_dim, d)
            self.state_mixing = nn.Parameter(torch.tensor(0.0))
        self.out_linear = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        B, N, _ = x.shape
        H, D = self.heads, self.dim_head
        x_mid = self.in_project_x(x).view(B, N, H, D)
        fx_mid = self.in_project_fx(x).view(B, N, H, D)
        w, tok = slice_tokens(x_mid, fx_mid, self.in_project_slice, self.temperature)

        q, k, v = self.qkv_project(tok).chunk(3, dim=-1)
        out = F.scaled_dot_product_attention(q, k, v)  # attention among the S slice tokens
        if context is not None:
            cross = F.scaled_dot_product_attention(
                self.cross_q(tok), self.cross_k(context), self.cross_v(context)
            )
            g = torch.sigmoid(self.state_mixing)
            out = g * out + (1 - g) * cross

        out = torch.einsum("bnhs,bhsd->bnhd", w, out).reshape(B, N, H * D)  # back to tokens
        return self.out_dropout(self.out_linear(out))


class GALEBlock(nn.Module):
    def __init__(self, dim, heads, slice_num, mlp_ratio, context_dim, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = GALE(dim, heads, slice_num, context_dim, dropout)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x, context=None):
        x = x + self.attn(self.ln_1(x), context)
        return x + self.mlp(self.ln_2(x))


class GeoTransolver(nn.Module):
    def __init__(
        self,
        functional_dim,
        out_dim,
        geometry_dim=None,
        global_dim=None,
        n_layers=4,
        n_hidden=256,
        n_head=8,
        slice_num=32,
        mlp_ratio=4,
        dropout=0.0,
    ):
        super().__init__()
        assert n_hidden % n_head == 0, "n_hidden must be divisible by n_head"
        d = n_hidden // n_head
        self.preprocess = nn.Sequential(
            nn.Linear(functional_dim, 2 * n_hidden), nn.GELU(), nn.Linear(2 * n_hidden, n_hidden)
        )
        self.geometry_tokenizer = ContextProjector(geometry_dim, n_head, d, slice_num) if geometry_dim else None
        self.global_tokenizer = ContextProjector(global_dim, n_head, d, slice_num) if global_dim else None
        context_dim = d * ((geometry_dim is not None) + (global_dim is not None))
        self.blocks = nn.ModuleList(
            GALEBlock(n_hidden, n_head, slice_num, mlp_ratio, context_dim, dropout) for _ in range(n_layers)
        )
        self.out = nn.Sequential(nn.LayerNorm(n_hidden), nn.Linear(n_hidden, out_dim))

    def build_context(self, geometry, global_embedding):
        parts = []
        if self.geometry_tokenizer is not None and geometry is not None:
            parts.append(self.geometry_tokenizer(geometry))
        if self.global_tokenizer is not None and global_embedding is not None:
            parts.append(self.global_tokenizer(global_embedding))
        return torch.cat(parts, dim=-1) if parts else None

    def forward(self, local_embedding, geometry=None, global_embedding=None):
        """local_embedding (B, N, C); geometry (B, N, C_geo); global_embedding (B, N_g, C_g)."""
        context = self.build_context(geometry, global_embedding)
        x = self.preprocess(local_embedding)
        for block in self.blocks:
            x = block(x, context)
        return self.out(x)
