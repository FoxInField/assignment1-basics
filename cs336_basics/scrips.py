from BPETokenizer import train_bpe, BPETokenizer
from cs336_basics.model import TransformerLM
from cs336_basics.Optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.Checkpointing import load_checkpoint
from cs336_basics.nn_untils import cross_entropy, gradient_clipping
from cs336_basics.data import get_batch
from tqdm import tqdm
import torch
import json
import os
import time
import cProfile
import pstats
import io
import numpy
import wandb

def train_volcabulary(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    output_path: str,
    **kwargs,
) -> BPETokenizer:
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    bpeTokenizer = BPETokenizer(vocab, merges, special_tokens)
    bpeTokenizer.save(output_path)
    print("vocabulary has save to: ", output_path)
    return bpeTokenizer

def preprocess(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    output_path: str,
    train_dataset_path: str,
    **kwargs,
):
    # 训练词汇表
    tokenizer = train_volcabulary(input_path, vocab_size, special_tokens, output_path)

    # 将文本转化为npy格式
    with open(input_path, "r", encoding="utf-8") as f:
        str = f.read()
    encoded_str = tokenizer.encode(str)
    data_array = numpy.array(encoded_str, dtype = numpy.int32)
    numpy.save(train_dataset_path, data_array)

if __name__ == "__main__":

    special_tokens = ["<|endoftext|>"]
    path_config = {
        "txt_input_path": "data/TinyStoriesV2-GPT4-valid.txt",
        "bpe_output_path": "data/vocab.json",
        "train_dataset_path": "data/TinyStoriesV2-GPT4-valid.npy",
        "checkpoint_load_path": ""
    }

    # 模型参数
    model_config = {
        "vocab_size": 10000,      # 词汇表大小
        "context_length": 256,    # 上下文长度
        "num_layers": 4,          # Transformer Block数
        "num_heads": 16,          # 注意力头数
        "d_model": 512,           # 嵌入空间维度
        "d_ff": 1344,             # 前馈网络维度
        "rope_theta": 10000,      # RoPE参数
    }

    # 优化器参数
    optim_config = {
        "lr": 3e-4,               # 学习率
        "weight_decay": 1e-2,     # 权重衰减
        "betas": (0.9, 0.999),    # AdamW的beta参数
        "max_norm": 1.0,          # 梯度裁剪的最大范数
    }

    # 训练参数
    train_config = {
        "batch_size": 16,         # 批次大小
        "total_epochs": 0.5,      # 训练轮数
        "checkpoint_freq": 2000,  # 每隔多少步保存一次检查点
        "log_freq": 10,           # 每隔多少步记录一次日志
        "val_freq": 400,          # 每隔多少步在验证集上评估
        "val_batch_size": 16,     # 验证时的批次大小
        "val_batches": 20,        # 验证时使用的批次数量
    }

    # # 初始化wandb
    # run = wandb.init(
    #     project="cs336-assignment-1",
    #     name="train_v1",
    #     config={
    #         "model": model_config,
    #         "optimizer": optim_config,
    #         "training": train_config,
    #     }
    # )
    
    # # 开始性能分析 
    # pr = cProfile.Profile()
    # pr.enable()  

    # 预处理
    # preprocess(path_config["txt_input_path"], model_config["vocab_size"], special_tokens, path_config["bpe_output_path"], path_config["train_dataset_path"])

    train_dataset = numpy.load(path_config["train_dataset_path"], mmap_mode = "r+")
    print("数据集加载完成")

    # 计算训练所需step
    total_tokens = train_dataset.shape[0]
    total_steps = int(train_config["total_epochs"] * total_tokens) // (train_config["batch_size"] * model_config["context_length"])
    print(f"总token数: {total_tokens}, 训练轮数: {train_config['total_epochs']}, batch大小: {train_config['batch_size']}, 上下文长度: {model_config['context_length']}")
    print(f"总训练步数: {total_steps}")


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # 初始化模型
    model = TransformerLM(
        vocab_size = model_config["vocab_size"], 
        context_length = model_config["context_length"], 
        d_model = model_config["d_model"], 
        num_layers = model_config["num_layers"], 
        num_heads = model_config["num_heads"], 
        d_ff = model_config["d_ff"], 
        rope_theta = model_config["rope_theta"],
        device = device
    )
    print("模型初始化完成")

    # 初始化优化器
    optimizer = AdamW(
        model.parameters(),
        lr=optim_config["lr"],
        weight_decay=optim_config["weight_decay"],
        betas=optim_config["betas"],
    )
    print("优化器初始化完成")

    # 如果有checkpoint，则加载checkpoint
    start_iter = 1
    if path_config["checkpoint_load_path"]:
        print(f"开始加载模型检查点: {path_config['checkpoint_load_path']}")
        start_iter = load_checkpoint(
            path_config["checkpoint_load_path"],
            model=model,
            optimizer=optimizer
        )
        start_iter += 1
        print(f"模型检查点加载成功，当前迭代次数: {start_iter}")
    else:
        print("没有提供模型检查点，开始从头训练。")

    for step in tqdm(range(start_iter, total_steps + 1), desc="训练进度", unit="step"):
        # 清空梯度
        optimizer.zero_grad()

        # 使用余弦退火更新学习率
        lr_now = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=optim_config["lr"],
            min_learning_rate=optim_config["lr"] * 0.01,
            warmup_iters=int(0.05 * total_steps),
            cosine_cycle_iters=total_steps,
        )

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_now
        
        # 获取batch数据
        inputs, targets = get_batch(
            train_dataset,
            batch_size = train_config["batch_size"],
            context_length = model_config["context_length"],
            device = device
        )

        # 前向传播
        logits = model(inputs)

        # 计算损失
        loss = cross_entropy(logits, targets)

        # 反向传播和优化参数
        loss.backward()

        # 梯度裁减
        gradient_clipping(model.parameters(), max_l2_norm=optim_config["max_norm"]) # 梯度裁剪
        optimizer.step()

        # 日志记录
        if step % train_config["log_freq"] == 0:
            print(f"Step {step}, Loss: {loss.item()}")

            # 使用wandb记录损失和梯度范数
            # wandb.log({"train_loss": loss.item(), "lr": lr_now, "step": step})
    
    # 停止分析
    # pr.disable()  

    # # 打印结果
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    # ps.print_stats()
    # print(s.getvalue())
