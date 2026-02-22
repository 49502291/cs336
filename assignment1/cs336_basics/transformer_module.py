import torch
import torch.nn as nn

from cs336_basics.selfattention_module import MultiHeadSelfAttention
from cs336_basics.ff_module import FF
from cs336_basics.rmsnorm_module import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope= None, device=None, dtype=None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ff = FF(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention block with residual connection and layer norm
        attn_output = self.attn(self.norm1(x), token_positions=token_positions)
        x = x + attn_output  # Residual connection
        ff_output = self.ff(self.norm2(x))
        x = x + ff_output  # Residual connection
        return x