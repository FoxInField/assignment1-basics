import torch
from jaxtyping import Float, Int
from torch import Tensor
from typing import Iterable, Iterator
from einops import einsum, rearrange

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
    # max_vals = torch.amax(inputs, dim=-1, keepdim=True)
    # shifted = inputs - max_vals

    # log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    # correct_logit = shifted[torch.arange(inputs.shape[0]), targets]

    # ce = -correct_logit + log_sum_exp
    # return ce.mean()
    # 原式：-log{ softmax(inputs)[targets] } 
    # 拆开softmax并化简：-logits[targets] + log(sum(exp(inputs)))
    
    # 对多维度输入reshape
    inputs_reshaped = rearrange(inputs, "b ... v -> (b ...) v")  # (batch_size, vocab_size)
    targets_reshaped = rearrange(targets, "b ... -> (b ...)")  # (batch_size,)

    # 对logits预处理，减去每个样本中的最大logit，防止上溢
    inputs_stable = inputs_reshaped - inputs_reshaped.max(dim=-1, keepdim=True).values

    # 计算交叉熵
    targets_logit = inputs_stable.gather(1, targets_reshaped.unsqueeze(1)).squeeze(1)
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs_stable), dim=-1))
    loss = -targets_logit + log_sum_exp
    # 平均交叉熵
    return loss.mean()


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