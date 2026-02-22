import torch
import torch.nn as nn
from cs336_basics.linear_module import Linear

class FF(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        Construct the SwiGLU feed-forward network.
        d_model: int Hidden dimension of the model
        d_ff: int Inner dimension of the feedforward layer
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # gate projection
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # down projection
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # value projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the SwiGLU feed-forward network. The input tensor x will have shape
        (batch_size, sequence_length, d_model). The output should have the same shape as x. 
        """
        output1 = self.w1(x)
        gate = output1 * torch.sigmoid(output1)
        value = self.w3(x)
        output2 = gate * value
        return self.w2(output2)