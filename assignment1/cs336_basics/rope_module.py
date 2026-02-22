import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()

        # Create the RoPE buffer freq shape (d_k / 2,) and postition shape (max_seq_len, )
        freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        # shape (seq_len, d_k / 2) where each column corresponds to a different frequency
        angles = torch.outer(positions, freq)

        # Precompute cos and sin, shape: (max_seq_len, d_k / 2)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        '''
        # Get the corresponding cos and sin values for the token positions
        # token_positions: (..., seq_len) -> cos/sin: (..., seq_len, d_k / 2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        # Split x into even and odd parts, each of shape (..., seq_len, d_k / 2)
        x_even = x[..., ::2]  # (..., seq_len, d_k / 2)
        x_odd = x[..., 1::2]  # (..., seq_len, d_k / 2)
        # Apply the RoPE transformation
        x_rotated_even = x_even * cos - x_odd * sin  # (..., seq_len, d_k / 2)
        x_rotated_odd = x_even * sin + x_odd * cos   # (..., seq_len, d_k / 2)
        # Interleave the even and odd parts back together, shape: (..., seq_len, d_k)
        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd
        return x_rotated