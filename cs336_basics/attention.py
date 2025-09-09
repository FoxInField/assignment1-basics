import torch
from torch import nn, Tensor
from einops import rearrange, einsum
from .nn_untils import softmax
from jaxtyping import Float, Int

def scaled_dot_product_attention(
    Q: torch.Tensor,  # (..., queries, d_k)
    K: torch.Tensor,  # (..., keys, d_k)
    V: torch.Tensor,  # (..., values, d_v)
    mask: torch.Tensor | None = None  # (..., queries, keys)
) -> torch.Tensor:  # (..., queries, d_v)
    """
    Compute Scaled Dot-Product Attention.
    """
    # 1. Scores = Q @ K.T  # 形状 [L_q, L_k]
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # 2. Scores_scaled = Scores / sqrt(d_k)
    dk = K.shape[-1]
    scores_scaled = scores / (dk ** 0.5)
    
    # 3. Apply mask (if provided)
    if mask is not None:
        scores_scaled = scores_scaled.masked_fill(mask == 0, float('-inf'))
    
    # 4. Attention = softmax(Scores_scaled, dim=-1)
    attention = softmax(scores_scaled, dim=-1)
    
    # 5. Output = Attention @ V  # 形状 [L_q, d_v]
    output = torch.matmul(attention, V)  # (..., queries, d_v)
    
    return output

def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    half_dim = d_k // 2
    # Compute inverse frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=in_query_or_key.dtype, device=in_query_or_key.device) / half_dim))
    
    # Compute angles: (..., seq_len, half_dim)
    angles = token_positions[..., None] * inv_freq[None, ...]  # broadcast
    
    # cos and sin: (..., seq_len, half_dim)
    cos = angles.cos()
    sin = angles.sin()
    
    # Split the last dimension into even and odd
    x1 = in_query_or_key[..., 0::2]
    x2 = in_query_or_key[..., 1::2]
    
    # Apply RoPE rotation
    x_rotated_1 = x1 * cos - x2 * sin
    x_rotated_2 = x1 * sin + x2 * cos
    
    # Interleave back
    out = torch.empty_like(in_query_or_key)
    out[..., 0::2] = x_rotated_1
    out[..., 1::2] = x_rotated_2
    
    return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Q/K/V/O 权重
        self.q_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.k_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.v_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.o_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))

        # 初始化
        nn.init.trunc_normal_(self.q_proj_weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        nn.init.trunc_normal_(self.k_proj_weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        nn.init.trunc_normal_(self.v_proj_weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        nn.init.trunc_normal_(self.o_proj_weight, mean=0.0, std=0.02, a=-0.04, b=0.04)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Linear projections
        Q = torch.matmul(x, self.q_proj_weight.t())  # [..., seq_len, d_model]
        K = torch.matmul(x, self.k_proj_weight.t())
        V = torch.matmul(x, self.v_proj_weight.t())

        # 2. Reshape for multi-head: (..., seq_len, d_model) -> (..., num_heads, seq_len, head_dim)
        Q = rearrange(Q, '... seq_len (h d) -> ... h seq_len d', h=self.num_heads)
        K = rearrange(K, '... seq_len (h d) -> ... h seq_len d', h=self.num_heads)
        V = rearrange(V, '... seq_len (h d) -> ... h seq_len d', h=self.num_heads)

        # 3. Optional causal mask
        seq_len = Q.shape[-2]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device), diagonal=1)

        # 4. Use scaled_dot_product_attention
        context_heads = scaled_dot_product_attention(Q, K, V, mask=~causal_mask)
        
        # 5. Merge heads: (..., h, seq_len, head_dim) -> (..., seq_len, d_model)
        context = rearrange(context_heads, "... h seq_len d -> ... seq_len (h d)")

        # 6. Final linear projection
        out = torch.matmul(context, self.o_proj_weight.t())  # (..., seq_len, d_model)

        return out

class MultiHeadSelfAttentionWithRoPE(nn.Module):
    """
    Multi-Head Self-Attention module with Rotary Positional Embedding (RoPE).
    """

    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device: torch.device | None = None, dtype=None):
        """
        Args:
            d_model (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            theta (float): Θ value for RoPE.
            max_seq_len (int): Maximum sequence length.
            device (torch.device | None): Device to store weights on.
            dtype (torch.dtype | None): Data type for weights.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len

        # Q/K/V/O projection weights
        self.q_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.k_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.v_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.o_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))

        # Weight initialization
        nn.init.trunc_normal_(self.q_proj_weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        nn.init.trunc_normal_(self.k_proj_weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        nn.init.trunc_normal_(self.v_proj_weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        nn.init.trunc_normal_(self.o_proj_weight, mean=0.0, std=0.02, a=-0.04, b=0.04)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_model).
            token_positions (torch.Tensor): Token positions of shape (..., seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_model).
        """
        # 1. Compute Q, K, V
        Q = torch.matmul(x, self.q_proj_weight.t())  # (..., seq_len, d_model)
        K = torch.matmul(x, self.k_proj_weight.t())
        V = torch.matmul(x, self.v_proj_weight.t())

        # 2. Reshape for multi-head
        Q = rearrange(Q, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads)
        K = rearrange(K, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads)
        V = rearrange(V, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads)

        # 3. Apply RoPE to Q and K
        Q = rope(self.head_dim, self.theta, self.max_seq_len, Q, token_positions)
        K = rope(self.head_dim, self.theta, self.max_seq_len, K, token_positions)

        # 4. Causal mask
        seq_len = Q.shape[-2]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device), diagonal=1)

        # 5. Compute attention
        context_heads = scaled_dot_product_attention(Q, K, V, mask=~causal_mask)

        # 6. Merge heads
        context = rearrange(context_heads, "... h seq_len d -> ... seq_len (h d)")

        # 7. Output projection
        out = torch.matmul(context, self.o_proj_weight.t())  # (..., seq_len, d_model)

        return out
