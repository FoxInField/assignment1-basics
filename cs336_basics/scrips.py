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
    chunk_size: int = 1024 * 1024,  # 每次读取1MB
    **kwargs,
):
    # 训练词汇表
    tokenizer = train_volcabulary(txt_input_path, vocab_size, special_tokens, bpe_output_path)

    # 处理训练集 - 分批读取和编码
    def process_large_file(file_path, output_path):
        all_encoded = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            buffer = ""
            while True:
                # 分批读取
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                buffer += chunk
                
                # 按行分割，处理完整的行
                lines = buffer.split('\n')
                # 保留最后不完整的行用于下一次读取
                buffer = lines.pop() if lines else ""
                
                # 编码完整的行
                for line in lines:
                    if line.strip():  # 跳过空行
                        encoded_line = tokenizer.encode(line + '\n')  # 加上换行符
                        all_encoded.extend(encoded_line)
        
        # 处理缓冲区中剩余的内容
        if buffer.strip():
            encoded_buffer = tokenizer.encode(buffer)
            all_encoded.extend(encoded_buffer)
        
        # 保存为numpy数组
        data_array = numpy.array(all_encoded, dtype=numpy.int32)
        numpy.save(output_path, data_array)
        return len(all_encoded)

    # 处理训练集和验证集
    print("处理训练集...")
    train_token_count = process_large_file(txt_input_path, train_dataset_path)
    print(f"训练集处理完成，共 {train_token_count} 个token")
    
    print("处理验证集...")
    valid_token_count = process_large_file(valid_txt_path, valid_dataset_path)
    print(f"验证集处理完成，共 {valid_token_count} 个token")

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

    # # 预处理
    # preprocess(
    #     txt_input_path = path_config["txt_input_path"], 
    #     valid_txt_path = path_config["valid_txt_pash"],
    #     vocab_size = model_config["vocab_size"], 
    #     special_tokens = special_tokens, 
    #     bpe_output_path = path_config["bpe_output_path"], 
    #     train_dataset_path = path_config["train_dataset_path"],
    #     valid_dataset_path = path_config["valid_dataset_path"]
    # )

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

def generate_text(
    prompts: str, 
    model: TransformerLM, 
    optimizer: AdamW, 
    max_length: int = 100, 
    temperature: float = 1.0,
    top_k: int = None  # 新增 top_k 参数
) -> str:
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
            
            # --- 新增 top-k 采样逻辑 ---
            if top_k is not None:
                # 获取top-k个概率最高的token和它们的索引
                top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
                
                # 从top-k中采样
                next_token_id = torch.multinomial(top_k_probs, num_samples=1).item()
                next_token_id = top_k_indices[next_token_id].item()
            else:
                # 原来的采样方式（从所有token中采样）
                next_token_id = torch.multinomial(probs, num_samples=1).item()
            # --- top-k 逻辑结束 ---
            
            # 将生成的token添加到序列中
            generated_ids.append(next_token_id)
            
            # 更新输入tensor
            input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)
            
            # 如果生成了结束符，停止生成
            if next_token_id in tokenizer.encode("<|endoftext|>"):
                break

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

    # 适用于 TinyStories 风格模型的测试 prompts 列表
    test_prompts = [
        # 1. 基础故事生成 (Basic Story Generation)
        "Hello, can you tell me a story?",
        "Tell me a story about a cat and a dog.",
        "Once upon a time, there was a little frog who was afraid of the water. One day...",
        "Write a story about a big tree.",

        # 2. 指令跟随与简单约束 (Instruction Following & Simple Constraints)
        "Write a short story where a boy finds a big, red ball. Use the words 'sunny', 'happy', and 'park'.",
        "Tell me a story about a robot, but the robot must be kind.",
        "Make a story with a dog, a bone, and a friend.",
        "Tell a story about a mouse. The story must have the word 'cheese' in it.",

        # 3. 简单问答与理解 (Simple Q&A & Comprehension) - 需要先提供上下文
        "Story: Tim the mouse found a shiny coin. He gave it to his friend Kim the bird. Kim was very happy. Question: Who found the coin?",
        "Story: The little fox was sad. It lost its toy. Then it saw the toy under the bed. Question: Why was the fox happy at the end?",
        "Story: Ben the bear ate too much honey. He got a tummy ache. Question: What should Ben not do next time?",

        # 4. 基本因果与逻辑 (Basic Cause & Effect)
        "Finish this story: The dog barked at the mailman. The mailman was scared. So he...",
        "Why was the girl sad in the story? Because she...",
        "What happened next? The boy planted a seed. He watered it every day. Then...",
        "The sky got dark. The wind blew hard. What will happen next?",

        # 5. 风格一致性 (Style Consistency)
        "The bunny hopped to the tree. It saw a bright leaf. The leaf was gold. The bunny took the leaf home. Write another sentence that fits this story.",
        "The cat sat on the mat. It was a soft mat. The sun was warm. Continue this story.",
        "Add one more sentence to this story: 'The dog ran in the park. It found a big stick.'"
    ]

    # 示例使用方式：循环测试所有 prompts
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i+1}: {prompt} ---")
        output = generate_text(prompts=prompt, model=model, optimizer=optimizer, max_length=200, temperature=0.9, top_k=50)
        print(f"Output: {output}")
    
    # 停止分析
    # pr.disable()  

    # # 打印结果
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    # ps.print_stats()
    # print(s.getvalue())
