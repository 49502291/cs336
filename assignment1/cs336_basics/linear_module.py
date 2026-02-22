import math
import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    """A linear transformation layer without bias: y = x @ W^T."""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # Truncated normal: mean=0, std=sqrt(2/(in+out)), truncated at ±3*std
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")
