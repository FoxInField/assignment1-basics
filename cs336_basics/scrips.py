from BPETokenizer import train_bpe, BPETokenizer
import json
import os
import time
import numpy

def train_volcabulary(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    output_path: str,
    **kwargs,
):
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    bpeTokenizer = BPETokenizer(vocab, merges, special_tokens)
    bpeTokenizer.save(output_path)
    print("vocabulary has save to: ", output_path)

if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    args = {
        "txt_input_path": "data/TinyStoriesV2-GPT4-train.txt",
        "vocab_size": 10000,
        "bpe_output_path": "data/vocab.json"
    }
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练词汇表
    train_volcabulary(args["txt_input_path"], args["vocab_size"], special_tokens, args["bpe_output_path"])
    
    # 记录结束时间并计算耗时
    end_time = time.time()
    training_time = end_time - start_time
    
    # 将时间转换为更易读的格式
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = training_time % 60
    
    print(f"训练耗时: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
