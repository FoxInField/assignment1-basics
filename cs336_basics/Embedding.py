import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, 
        device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device = device, dtype = dtype))
        nn.init.trunc_normal_(self.weights, mean=0.0, std=0.02, a=-0.04, b=0.04)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]
