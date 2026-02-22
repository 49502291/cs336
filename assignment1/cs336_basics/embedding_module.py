import torch
import torch.nn as nn
from einops import einsum

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        ''' Construct an embedding module. 
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weights = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        # Truncated normal: mean=0, std=1, truncated at ±3*std
        nn.init.trunc_normal_(self.weights, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''
        Uses PyTorch's advanced indexing — each token ID selects the corresponding row
        from the (num_embeddings, embedding_dim) weight matrix, producing output shape
        (batch_size, sequence_length, embedding_dim)       
        '''
        return self.weights[token_ids]