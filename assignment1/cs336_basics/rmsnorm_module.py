import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. This function should accept the following parameters:
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.eps = eps
        # shape 1D vector (d_model,) learnable scaling parameter initialized to ones
        self.scale = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to the input tensor x. The input tensor x will have shape
        (batch_size, sequence_length, d_model). The output should have the same shape as x.
        """
        in_dtype = x.dtype
        # Ensure computations are done in float32 for numerical stability, then cast back to input dtype
        x = x.to(torch.float32)
        # Compute the root mean square of the last dimension (d_model)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize x by dividing by the RMS and then scaling by the learnable parameter
        result = x / rms * self.scale
        return result.to(in_dtype)