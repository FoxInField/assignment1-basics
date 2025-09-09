import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float, Int
from .RMSNorm import RMSNorm
from .attention import MultiHeadSelfAttentionWithRoPE
from .SwiGLU_FFN import SwiGLU_FFN
from .Embedding import Embedding
from .Linear import Linear

class TransformerBlock(nn.Module):
    """
    Transformer块，包含多头自注意力和前馈网络
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, d_ff: int = None, theta: float=10000.0, device: torch.device | None = None):
        """
        初始化Transformer块
        Args:
            d_model: 输入特征的维度
            num_heads: 多头注意力的头数
            d_ff: 前馈网络的中间层维度
            max_seq_len: 最大序列长度，位置嵌入用
            theta: 旋转位置编码的基数
            
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.attention = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, theta, max_seq_len, device=device)
        self.ffn = SwiGLU_FFN(d_model, d_ff, device = device)
        self.norm1 = RMSNorm(d_model, device=device)
        self.norm2 = RMSNorm(d_model, device=device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状为(batch, ..., seq_len, d_model)
            token_positions: 位置索引，形状为(batch, ..., seq_len)
        Returns:
            输出张量，形状为(batch, ..., seq_len, d_model)
        """
        token_positions = torch.arange(x.shape[-2], dtype=torch.int, device=x.device)  # (batch, ..., seq_len)

        # 多头自注意力
        attn_output = self.attention(
            self.norm1(x), token_positions
        )
        x2 = x + attn_output
        # 前馈网络
        ffn_output = self.ffn(self.norm2(x2))
        return x2 + ffn_output

class TransformerLM(nn.Module):
    """
    Transformer语言模型
    """
    def __init__(self, 
                 vocab_size: int, 
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int = None,
                 rope_theta: float = 10000.0,
                 device: torch.device | None = None,
                 dtype=None):
        """
        初始化Transformer语言模型
        Args:
            vocab_size: 词汇表大小
            context_length: 上下文长度，也就是最大序列长度
            d_model: 输入特征的维度
            num_heads: 多头注意力的头数
            num_layers: Transformer块的层数
            d_ff: 前馈网络的中间层维度
            rope_theta: 旋转位置编码的基数
        """
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model, device=device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, context_length, d_ff, rope_theta, device=device)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device)
        self.output_embedding = Linear(d_model, vocab_size, device=device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状为(batch, seq_len)，每个元素是词汇表中的索引
        Returns:
            输出张量，形状为(batch, seq_len, vocab_size)
        """
        # 嵌入
        x = self.token_embedding(x)
        # Transformer块
        for block in self.layers:
            x = block(x)
        # 最终归一化
        x = self.ln_final(x)
        # 输出层
        x = self.output_embedding(x)
        # # softmax，暂时先不用
        # return softmax(x, dim=-1)
        return x