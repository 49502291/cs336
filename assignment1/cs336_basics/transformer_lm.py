import torch
import torch.nn as nn

from cs336_basics.embedding_module import Embedding
from cs336_basics.transformer_module import TransformerBlock
from cs336_basics.rmsnorm_module import RMSNorm
from cs336_basics.linear_module import Linear
from cs336_basics.rope_module import RotaryPositionalEmbedding

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        if rope_theta is not None:
            rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length, device=device)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, rope=rope, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, vocab_size, device=device, dtype=dtype)


    # output shape (batch_size, sequence_length, vocab_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(x.shape[1], device=x.device)  # (seq_len,)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.norm(x)
        logits = self.output_proj(x)
        return logits