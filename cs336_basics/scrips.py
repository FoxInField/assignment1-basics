from BPETokenizer import train_bpe, BPETokenizer
from cs336_basics.model import TransformerLM
from cs336_basics.Optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.Checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.nn_untils import cross_entropy, gradient_clipping, softmax, evaluate_model
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

special_tokens = ["<|endoftext|>"]
path_config = {
    "txt_input_path": "data/TinyStoriesV2-GPT4-train.txt",
    "valid_txt_pash": "data/TinyStoriesV2-GPT4-valid.txt",
    "bpe_output_path": "data/vocab.json",
    "train_dataset_path": "data/TinyStoriesV2-GPT4-train.npy",
    "valid_dataset_path": "data/TinyStoriesV2-GPT4-valid.npy",
    "checkpoint_load_path": "data/weights.pt"
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
    "lr": 1e-4,               # 学习率
    "weight_decay": 1e-2,     # 权重衰减
    "betas": (0.9, 0.999),    # AdamW的beta参数
    "max_norm": 1.0,          # 梯度裁剪的最大范数
}

# 训练参数
train_config = {
    "batch_size": 16,         # 批次大小
    "total_epochs": 1,      # 训练轮数
    "checkpoint_freq": 2000,  # 每隔多少步保存一次检查点
    "log_freq": 10,           # 每隔多少步记录一次日志
    "val_freq": 400,          # 每隔多少步在验证集上评估
    "val_batch_size": 16,     # 验证时的批次大小
    "val_batches": 20,        # 验证时使用的批次数量
}

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
    txt_input_path: str | os.PathLike,
    valid_txt_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    bpe_output_path: str,
    train_dataset_path: str,
    valid_dataset_path: str,
    **kwargs,
):
    # 训练词汇表
    tokenizer = train_volcabulary(txt_input_path, vocab_size, special_tokens, bpe_output_path)

    # 将文本转化为npy格式
    with open(txt_input_path, "r", encoding="utf-8") as f:
        str = f.read()
    encoded_str = tokenizer.encode(str)
    data_array = numpy.array(encoded_str, dtype = numpy.int32)
    numpy.save(train_dataset_path, data_array)

    # 将文本转化为npy格式
    with open(valid_txt_path, "r", encoding="utf-8") as f:
        str = f.read()
    encoded_str = tokenizer.encode(str)
    data_array = numpy.array(encoded_str, dtype = numpy.int32)
    numpy.save(valid_dataset_path, data_array)

def train(model: TransformerLM, optimizer: AdamW):
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

    # 预处理
    preprocess(
        txt_input_path = path_config["txt_input_path"], 
        valid_txt_path = path_config["valid_txt_pash"],
        vocab_size = model_config["vocab_size"], 
        special_tokens = special_tokens, 
        bpe_output_path = path_config["bpe_output_path"], 
        train_dataset_path = path_config["train_dataset_path"],
        valid_dataset_path = path_config["valid_dataset_path"]
    )

    train_dataset = numpy.load(path_config["train_dataset_path"], mmap_mode = "r+")
    validation_dataset = numpy.load(path_config["valid_dataset_path"], mmap_mode = "r+")
    print("数据集加载完成")

    # 计算训练所需step
    total_tokens = train_dataset.shape[0]
    total_steps = int(train_config["total_epochs"] * total_tokens) // (train_config["batch_size"] * model_config["context_length"])
    print(f"总token数: {total_tokens}, 训练轮数: {train_config['total_epochs']}, batch大小: {train_config['batch_size']}, 上下文长度: {model_config['context_length']}")
    print(f"总训练步数: {total_steps}")

    # 如果有checkpoint，则加载checkpoint
    start_iter = 1
    if path_config["checkpoint_load_path"] and os.path.isfile(path_config["checkpoint_load_path"]):
        print(f"开始加载模型检查点: {path_config['checkpoint_load_path']}")
        start_iter = load_checkpoint(
            path_config["checkpoint_load_path"],
            model=model,
            optimizer=optimizer
        )
        start_iter += 1
        print(f"模型检查点加载成功，当前迭代次数: {start_iter}")
    elif path_config["checkpoint_load_path"]:
        print(f"警告: 检查点文件不存在: {path_config['checkpoint_load_path']}，开始从头训练。")
    else:
        print("没有提供模型检查点路径，开始从头训练。")

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
        
        # 在验证集上评估模型
        if validation_dataset is not None and step % train_config["val_freq"] == 0:
            val_loss = evaluate_model(
                model=model,
                dataset=validation_dataset,
                device=device,
                batch_size=train_config["val_batch_size"],
                context_length=model_config["context_length"],
                num_batches=train_config["val_batches"]
            )
            print(f"验证集损失: {val_loss}")

    save_checkpoint(model = model, optimizer = optimizer, iteration = start_iter, out = path_config["checkpoint_load_path"])
    print(f"save checkpoint to {path_config['checkpoint_load_path']}")

def generate_text(prompts: str, model: TransformerLM, optimizer: AdamW, max_length: int = 100, temperature: float = 1.0) -> str:
    load_checkpoint(
        path_config["checkpoint_load_path"],
        model=model,
        optimizer=optimizer
    )
    model.eval()

    tokenizer = BPETokenizer.load(path_config["bpe_output_path"])

    # 编码初始prompt
    input_ids = tokenizer.encode(prompts)
    generated_ids = input_ids.copy()

     # 转换为tensor并移动到设备
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # 自回归生成
    with torch.no_grad():  # 禁用梯度计算
        for _ in range(max_length):
            # 前向传播
            logits = model(input_tensor)
            
            # 只取最后一个时间步的logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # 应用softmax得到概率分布
            probs = softmax(next_token_logits, dim=-1)
            
            # 从概率分布中采样下一个token
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # 将生成的token添加到序列中
            generated_ids.append(next_token_id)
            
            # 更新输入tensor
            input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)
            
            # 如果生成了结束符，停止生成
            if next_token_id in tokenizer.encode("<|endoftext|>"):break

    output = tokenizer.decode(generated_ids)
    return output


if __name__ == "__main__":
    # # 开始性能分析 
    # pr = cProfile.Profile()
    # pr.enable()  

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

    # train(model=model, optimizer=optimizer)

    prompts = "Hello, can you tell me a story?"
    output = generate_text(prompts=prompts, model=model, optimizer=optimizer)
    print(output)
    
    # 停止分析
    # pr.disable()  

    # # 打印结果
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    # ps.print_stats()
    # print(s.getvalue())
