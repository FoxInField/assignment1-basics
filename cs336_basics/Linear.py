import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weights, mean=0.0, std=0.02, a=-0.04, b=0.04)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, 
            '... in_features, out_features in_features -> ... out_features')
