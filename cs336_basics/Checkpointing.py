import torch
from typing import Union, BinaryIO, IO
import os

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]],
):
    """
    Save model state, optimizer state, and iteration to a checkpoint file.
    """
    # 将要保存的内容打包成字典
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    # 使用 torch.save 保存到指定路径或文件对象
    torch.save(checkpoint, out)


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load model state, optimizer state, and iteration from a checkpoint file.
    Returns:
        iteration (int): the saved iteration number.
    """
    # 从文件加载checkpoint
    checkpoint = torch.load(src)

    # 恢复模型和优化器状态
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    # 返回保存的迭代数
    return checkpoint["iteration"]
