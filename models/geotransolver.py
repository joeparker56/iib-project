"""
The GeoTransolver architecture works using a transformer architecture.

It embeds the points as physically sensical embeddings, and then embeds the geomtry by using a cross-attention mechanism.

"""

import math
import torch 
import torch.nn as nn
import torch.nn.functional as F


class ContextProjector(nn.Module):
    """
    Compress variable-length point features into S slice tokens.
    each batch (B) has N points, with input dim 
    Input:  [B, N, input_dim]
    Output: [B, H, S, D]
    """

    def __init__(self, input_dim, heads=4, slices=16, head_dim=32):
        super().__init__()
        self.heads = heads
        self.slices = slices
        self.head_dim = head_dim

        self.q = nn.Linear(input_dim, heads * head_dim) #query vectors
        self.k = nn.Linear(input_dim, heads * head_dim) #key vectors 
        self.slice_proj = nn.Parameter(
            torch.randn(heads, head_dim, slices) * 0.02
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        B, N, _ = x.shape
        H, D = self.heads, self.head_dim

        q = self.q(x).reshape(B, N, H, D).transpose(1, 2)
        k = self.k(x).reshape(B, N, H, D).transpose(1, 2)

        logits = torch.einsum("bhnd,hds->bhns", q, self.slice_proj)

        tau = self.temperature.clamp_min(0.05)
        weights = torch.softmax(logits / tau, dim=-1)

        # Normalise each slice across contributing points
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-6)

        return torch.einsum("bhns,bhnd->bhsd", weights, k)


# -------------------------------------------------------
# 2. Multi-scale geometric neighbourhood encoder
# -------------------------------------------------------

@torch.no_grad()
def ball_query(query, source, radius, k, max_elements=2**26):
    """
    For every query point, find up to k source points within `radius`,
    ordered by increasing distance.

    query:  [B, N, 3]
    source: [B, M, 3]

    Returns:
        idx:  [B, N, k] long  - index into source (0 where no neighbour)
        mask: [B, N, k] bool  - True where the slot holds a real neighbour

    The search is chunked over query points so that the pairwise distance
    block never exceeds `max_elements` entries, keeping memory bounded
    even for clouds with 1e5+ points on each side.
    """
    B, N, _ = query.shape
    M = source.shape[1]
    k_actual = min(k, M)

    # Largest query chunk whose [B, chunk, M] distance block fits the budget
    chunk = max(1, max_elements // (B * M))

    idx = torch.zeros(B, N, k, dtype=torch.long, device=query.device)
    mask = torch.zeros(B, N, k, dtype=torch.bool, device=query.device)

    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        dist = torch.cdist(query[:, start:stop], source)  # [B, c, M]

        # Push everything outside the ball to +inf so topk only
        # returns true ball members (inf entries are flagged by the mask)
        dist = dist.masked_fill(dist > radius, float("inf"))
        d, i = dist.topk(k_actual, dim=-1, largest=False)

        idx[:, start:stop, :k_actual] = i
        mask[:, start:stop, :k_actual] = torch.isfinite(d)

    return idx, mask


class GeometryEncoder(nn.Module):
    """
    Geometric processor: for each query point, ball-query up to K
    geometry points within a specified radius.

    Relative coordinates -> MLP -> geometric embedding.

    Following the PointNet++ convention, empty neighbour slots are
    filled with the first valid neighbour (or zeros if the ball is empty)
    so that padding is never mistaken for a point coincident with the query.
    """

    def __init__(self, radius, k=8, hidden=32):
        super().__init__()
        self.radius = radius
        self.k = k

        self.mlp = nn.Sequential(
            nn.Linear(3 * k, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.Tanh()
        )

    def forward(self, query, source):
        # query:  [B, N, 3]  points we want an embedding for
        # source: [B, M, 3]  points being searched
        B, N, _ = query.shape

        idx, mask = ball_query(query, source, self.radius, self.k)

        # Replace empty slots with the first (nearest) valid neighbour.
        # Results are sorted by distance, so slot 0 is valid whenever
        # the ball contains anything at all.
        idx = torch.where(mask, idx, idx[..., :1])
        any_valid = mask[..., :1]

        batch = torch.arange(B, device=query.device)[:, None, None]
        neighbours = source[batch, idx]  # [B, N, K, 3]

        relative = neighbours - query.unsqueeze(2)
        relative = relative * any_valid.unsqueeze(-1)

        relative = relative.reshape(B, N, 3 * self.k)

        return self.mlp(relative)


# -------------------------------------------------------
# 3. GALE: Geometry-Aware Latent Embedding
# -------------------------------------------------------

class GALE(nn.Module):
    """
    Transolver-style physics attention with persistent
    geometry cross-attention.

    x:       [B, N, hidden]
    context: [B, H, S, context_dim]
    """

    def __init__(
        self, hidden=128, heads=4, slices=16,
        context_dim=160
    ):
        super().__init__()

        assert hidden % heads == 0

        self.heads = heads
        self.slices = slices
        self.head_dim = hidden // heads

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        self.slice_logits = nn.Linear(hidden, heads * slices)
        self.value = nn.Linear(hidden, hidden)

        self.self_attention = nn.MultiheadAttention(
            self.head_dim, num_heads=1, batch_first=True
        )

        self.context_key = nn.Linear(context_dim, self.head_dim)
        self.context_value = nn.Linear(context_dim, self.head_dim)

        self.cross_attention = nn.MultiheadAttention(
            self.head_dim, num_heads=1, batch_first=True
        )

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.output_proj = nn.Linear(hidden, hidden)

        self.ffn = nn.Sequential(
            nn.Linear(hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden)
        )

    def forward(self, x, context):
        B, N, C = x.shape
        H, S = self.heads, self.slices
        D = self.head_dim

        z = self.norm1(x)

        # Assign every point to learned physics slices
        weights = self.slice_logits(z)
        weights = weights.reshape(B, N, H, S)
        weights = weights.permute(0, 2, 1, 3)
        weights = torch.softmax(weights, dim=-1)

        # Aggregate point features into slice tokens
        values = self.value(z).reshape(B, N, H, D)
        values = values.permute(0, 2, 1, 3)

        normalised = weights / weights.sum(
            dim=2, keepdim=True
        ).clamp_min(1e-6)

        slices = torch.einsum(
            "bhns,bhnd->bhsd", normalised, values
        )

        # Treat each head as an independent attention batch
        slices_flat = slices.reshape(B * H, S, D)

        # Self-attention between physics slices
        self_out, _ = self.self_attention(
            slices_flat, slices_flat, slices_flat,
            need_weights=False
        )

        # Cross-attention to persistent geometric context
        keys = self.context_key(context).reshape(B * H, S, D)
        vals = self.context_value(context).reshape(B * H, S, D)

        cross_out, _ = self.cross_attention(
            slices_flat, keys, vals, need_weights=False
        )

        # Learnable balance between the two attention paths
        alpha = torch.sigmoid(self.alpha)
        mixed = (1 - alpha) * self_out + alpha * cross_out
        mixed = mixed.reshape(B, H, S, D)

        # De-slice: map latent tokens back to input points
        points = torch.einsum("bhns,bhsd->bhnd", weights, mixed)
        points = points.permute(0, 2, 1, 3).reshape(B, N, C)

        x = x + self.output_proj(points)
        x = x + self.ffn(self.norm2(x))

        return x
class GeoTransolver(nn.Module):

    def __init__(
        self,
        feature_dim=1,
        param_dim=2,
        output_dim=4,
        hidden=128,
        heads=4,
        slices=16,
        layers=4,
        geo_hidden=32,
        radii=(0.1, 0.4, 1.0),
        k=8,
        global_dim=0
    ):
        super().__init__()

        assert hidden % heads == 0

        D = hidden // heads
        self.radii = radii

        # Geometry encoders in both directions
        self.local_encoders = nn.ModuleList([
            GeometryEncoder(r, k, geo_hidden) for r in radii
        ])

        self.context_encoders = nn.ModuleList([
            GeometryEncoder(r, k, geo_hidden) for r in radii
        ])

        # Separate context projectors for each source
        self.geometry_projector = ContextProjector(
            3, heads, slices, D
        )

        self.global_projector = ContextProjector(
            param_dim, heads, slices, D
        )

        self.scale_projectors = nn.ModuleList([
            ContextProjector(geo_hidden, heads, slices, D)
            for _ in radii
        ])

        # Context sources are concatenated along features
        context_dim = (2 + len(radii)) * D

        # Query embedding + local geometry augmentation
        self.input_proj = nn.Sequential(
            nn.Linear(3 + feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden)
        )

        self.local_proj = nn.Linear(
            hidden + len(radii) * geo_hidden, hidden
        )

        # Repeated GALE layers
        self.blocks = nn.ModuleList([
            GALE(hidden, heads, slices, context_dim)
            for _ in range(layers)
        ])

        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim)
        )
        self.global_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, global_dim)
        ) if global_dim else None
        

    def forward(self, x, features, geometry, params):
        """
        x:        [B, N, 3]  query positions
        features: [B, N, F]  point features
        geometry: [B, M, 3]  surface point cloud
        params:   [B, P]     global operating parameters
        """

        B = x.shape[0]

        # ---- Local geometry information ----
        local_features = [
            enc(x, geometry)
            for enc in self.local_encoders
        ]
        local = torch.cat(local_features, dim=-1)

        # ---- Shared context ----
        contexts = [
            self.geometry_projector(geometry),
            self.global_projector(params.unsqueeze(1))
        ]

        for enc, projector in zip(
            self.context_encoders, self.scale_projectors
        ):
            # Geometry points query nearby field points
            geo_features = enc(geometry, x)
            contexts.append(projector(geo_features))

        context = torch.cat(contexts, dim=-1)

        # ---- Input token embedding ----
        h = self.input_proj(torch.cat([x, features], dim=-1))
        h = self.local_proj(torch.cat([h, local], dim=-1))

        # ---- Persistent geometry-aware physics attention ----
        for block in self.blocks:
            h = block(h, context)

        fields = self.output_head(h)
        if self.global_head is None:
            return fields
        return fields, self.global_head(h.mean(dim=1))


# -------------------------------------------------------
# 5. Minimal runnable test
# -------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    model = GeoTransolver(
        feature_dim=1,
        param_dim=2,
        output_dim=4,
        hidden=64,
        heads=4,
        slices=16,
        layers=3
    )

    B, N, M = 2, 128, 64

    x = torch.rand(B, N, 3)
    features = torch.rand(B, N, 1)
    geometry = torch.rand(B, M, 3)
    params = torch.tensor([
        [0.3, 5.0],
        [0.5, 2.0]
    ])

    predictions = model(x, features, geometry, params)

    print("Predictions:", predictions.shape)

    # Check gradients work throughout the network
    loss = predictions.square().mean()
    loss.backward()

    print("Loss:", loss.item())
    print("Backward pass successful")
