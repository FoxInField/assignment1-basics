import torch
from jaxtyping import Float, Int
from torch import Tensor
from typing import Iterable, Iterator

def softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor with the same shape as `in_features` with softmax applied along `dim`.
    """
    shifted = in_features - torch.amax(in_features, dim=dim, keepdim=True)
    exp_vals = torch.exp(shifted)
    softmax_vals = exp_vals / torch.sum(exp_vals, dim=dim, keepdim=True)
    return softmax_vals

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    max_vals = torch.amax(inputs, dim=-1, keepdim=True)
    shifted = inputs - max_vals

    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    correct_logit = shifted[torch.arange(inputs.shape[0]), targets]

    ce = -correct_logit + log_sum_exp
    return ce.mean()

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # 1. 收集所有非 None 梯度
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    # 2. 计算总 L2 范数
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))

    # 3. 计算裁剪系数
    clip_coef = max_l2_norm / (total_norm + 1e-32)  # 加 epsilon 防止除零
    clip_coef = min(1.0, clip_coef)  # 不放大梯度，只缩小

    # 4. 原地修改梯度
    for g in grads:
        g.data.mul_(clip_coef)