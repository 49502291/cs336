import math
import torch
import torch.nn as nn
from einops import einsum, rearrange

from cs336_basics.linear_module import Linear


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax to the i-th dimension of the input tensor.

    softmax(x_i) = exp(x_i) / sum(exp(x_j))

    For numerical stability, subtract the max before exponentiating:
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    Args:
        x: Input tensor of arbitrary shape.
        dim: The dimension along which to apply softmax.

    Returns:
        Tensor of the same shape as x with softmax applied along dim.
    """
    # Subtract max for numerical stability (prevents overflow in exp)
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Args:
        Q: Query tensor of shape (..., seq_len, d_k)
        K: Key tensor of shape (..., seq_len, d_k)
        V: Value tensor of shape (..., seq_len, d_v)
        mask: Optional boolean mask of shape (..., seq_len, seq_len).
              False = position is masked (set to -inf before softmax).

    Returns:
        Output tensor of shape (..., seq_len, d_v)
    """
    d_k = Q.shape[-1]
    # Q K^T / sqrt(d_k), shape: (..., seq_len, seq_len)
    attn_scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    # Apply mask: set masked positions to -inf so softmax gives them 0 weight
    if mask is not None:
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
    # Softmax over keys dimension, shape: (..., seq_len, seq_len)
    attn_weights = softmax(attn_scores, dim=-1)
    # Weighted sum of values, shape: (..., seq_len, d_v)
    return einsum(attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: nn.Module | None = None, device=None, dtype=None):
        """
        Causal multi-head self-attention.

        Args:
            d_model: Dimensionality of the input and output.
            num_heads: Number of attention heads. d_model must be divisible by num_heads.
            rope: Optional RoPE module to apply to Q and K.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # head dimension

        # Combined Q, K, V projections: each (d_model -> d_model)
        # Conceptually num_heads separate (d_model -> d_k) projections stacked
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        # Output projection: (d_model -> d_model)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            token_positions: Optional position indices of shape (batch_size, seq_len)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V — each (batch_size, seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to (batch_size, num_heads, seq_len, d_k)
        Q = rearrange(Q, "batch seq (head dk) -> batch head seq dk", head=self.num_heads)
        K = rearrange(K, "batch seq (head dk) -> batch head seq dk", head=self.num_heads)
        V = rearrange(V, "batch seq (head dk) -> batch head seq dk", head=self.num_heads)

        # Apply RoPE to Q and K if provided
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Create causal mask: True = allowed to attend, False = masked
        # Each query at position i can attend to keys at positions <= i
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        # Scaled dot-product attention with causal mask
        # Q, K, V: (batch_size, num_heads, seq_len, d_k)
        # causal_mask: (seq_len, seq_len) — broadcasts over batch and heads
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # Reshape back: (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        attn_output = rearrange(attn_output, "batch head seq dk -> batch seq (head dk)", head=self.num_heads)

        # Output projection
        return self.output_proj(attn_output)