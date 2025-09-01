import torch
from torch import Tensor
from jaxtyping import Float, Int
from .RMSNorm import RMSNorm
from .attention import MultiHeadSelfAttentionWithRoPE
from .SwiGLU_FFN import SwiGLU_FFN
from .Embedding import Embedding
from .Linear import Linear

def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    eps = 1e-5
    batch_size, sequence_length, d_model = in_features.shape
    token_positions = torch.arange(sequence_length, device=in_features.device)  # [0..sequence_length-1]
    token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)       # [batch, sequence_length] 

    norm1 = RMSNorm(d_model, eps, in_features.device, in_features.dtype)
    norm1.weight.data.copy_(weights['ln1.weight'])

    multihead_self_attention_with_rope = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, theta, max_seq_len, in_features.device, in_features.dtype)
    multihead_self_attention_with_rope.q_proj_weight.data.copy_(weights['attn.q_proj.weight'])
    multihead_self_attention_with_rope.k_proj_weight.data.copy_(weights['attn.k_proj.weight'])
    multihead_self_attention_with_rope.v_proj_weight.data.copy_(weights['attn.v_proj.weight'])
    multihead_self_attention_with_rope.o_proj_weight.data.copy_(weights['attn.output_proj.weight'])


    norm2 = RMSNorm(d_model, eps, in_features.device, in_features.dtype)
    norm2.weight.data.copy_(weights['ln2.weight'])

    ffn = SwiGLU_FFN(d_model, d_ff)
    ffn.w1.weights.data.copy_(weights['ffn.w1.weight'])
    ffn.w2.weights.data.copy_(weights['ffn.w2.weight'])
    ffn.w3.weights.data.copy_(weights['ffn.w3.weight'])

    in_features = in_features + multihead_self_attention_with_rope(norm1(in_features), token_positions)
    output = in_features + ffn(norm2(in_features))
    return output
    
def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    eps = 1e-5
    batch_size, sequence_length = in_indices.shape
    device = in_indices.device
    dtype = torch.float32

    token_positions = torch.arange(sequence_length, device=device)
    token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)

    embedding_layer = Embedding(vocab_size, d_model, device=device, dtype=dtype)
    embedding_layer.weights.data.copy_(weights['token_embeddings.weight'].to(dtype=dtype))
    x = embedding_layer(in_indices)

    for layer_idx in range(num_layers):
        layer_weights = {
            'attn.q_proj.weight': weights[f'layers.{layer_idx}.attn.q_proj.weight'].to(dtype=dtype),
            'attn.k_proj.weight': weights[f'layers.{layer_idx}.attn.k_proj.weight'].to(dtype=dtype),
            'attn.v_proj.weight': weights[f'layers.{layer_idx}.attn.v_proj.weight'].to(dtype=dtype),
            'attn.output_proj.weight': weights[f'layers.{layer_idx}.attn.output_proj.weight'].to(dtype=dtype),
            'ln1.weight': weights[f'layers.{layer_idx}.ln1.weight'].to(dtype=dtype),
            'ffn.w1.weight': weights[f'layers.{layer_idx}.ffn.w1.weight'].to(dtype=dtype),
            'ffn.w2.weight': weights[f'layers.{layer_idx}.ffn.w2.weight'].to(dtype=dtype),
            'ffn.w3.weight': weights[f'layers.{layer_idx}.ffn.w3.weight'].to(dtype=dtype),
            'ln2.weight': weights[f'layers.{layer_idx}.ln2.weight'].to(dtype=dtype),
        }

        x = transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=x,
        )

    final_norm = RMSNorm(d_model, eps, device=device, dtype=dtype)
    final_norm.weight.data.copy_(weights['ln_final.weight'].to(dtype=dtype))
    x = final_norm(x)

    lm_head = Linear(d_model, vocab_size, device, dtype)
    lm_head.weights.data.copy_(weights['lm_head.weight'].to(dtype=dtype))
    output = lm_head(x) 

    return output