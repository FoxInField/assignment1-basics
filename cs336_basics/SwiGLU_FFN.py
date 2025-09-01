import torch
import torch.nn as nn
from cs336_basics.Linear import Linear

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x) # sigmoid(x) = 1 / (1 + exp(-x))

class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # 三个线性层
        self.w1 = Linear(d_model, d_ff)
        self.w3 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU 公式: out = (SiLU(x W1) * (x W3)) W2
        x1 = self.w1(x)
        x2 = self.w3(x)
        gated = silu(x1) * x2
        out = self.w2(gated)
        return out
