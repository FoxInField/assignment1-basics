from BPETokenizer import train_bpe, BPETokenizer
import json
import os

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
        "vocab_size": 5000,
        "bpe_output_path": "data/vocab.json"
    }
    train_volcabulary(args["txt_input_path"], args["vocab_size"], special_tokens, args["bpe_output_path"])

    # tokenizer = BPETokenizer.load(args["bpe_output_path"])
