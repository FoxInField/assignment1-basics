from collections.abc import Callable, Iterable
from typing import Optional
import torch
from torch.optim import Optimizer
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.

        for p in group["params"]:
            if(p.grad) is None:
                continue

            state = self.state[p]
            t = state.get("t", 0)
            grad = p.grad.data
            p.data -= lr / math.sqrt(t + 1) * grad
            state["t"] = t + 1
        
        return loss

class AdamW(Optimizer):
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0, amsgrad=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # Get state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1
                step = state['step']

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected estimates
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                if amsgrad:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = lr / bias_correction1

                # Decoupled weight decay
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

                # Parameter update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Compute learning rate with linear warmup and cosine decay.

    Args:
        it (int): Current iteration number.
        max_learning_rate (float): alpha_max.
        min_learning_rate (float): alpha_min.
        warmup_iters (int): T_w, warmup iterations.
        cosine_cycle_iters (int): T_c, cosine decay iterations.

    Returns:
        float: Learning rate at iteration `it`.
    """
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it <= cosine_cycle_iters:
        t = it - warmup_iters
        cosine_decay = 0.5 * (1 + math.cos(math.pi * t / (cosine_cycle_iters - warmup_iters)))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
    else:
        return min_learning_rate
    
if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))

    # 定义优化器
    opt = SGD([weights], lr=5)

    # 训练循环
    for t in range(100):
        opt.zero_grad()                 # 清空梯度
        loss = (weights ** 2).mean()    # 计算标量损失
        print(loss.item())              # 打印损失
        loss.backward()                 # 反向传播计算梯度
        opt.step()       
