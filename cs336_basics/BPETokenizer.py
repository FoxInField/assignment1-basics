import os
import regex as re
from typing import BinaryIO, Tuple, Set, DefaultDict, List, Dict
from collections import defaultdict

def split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    Split on the special tokens
    example: 
        text = "Hello world! <|endoftext|> Great!" 
        special_tokens = "<|endoftext|>"
        result = ['Hello world! ', '<|endoftext|>', ' Great!']
    """
    special_tokens_sorted = sorted(special_tokens, key=lambda x: -len(x))
    if not special_tokens_sorted:
        parts = [text]
    else:
        pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)
        parts = re.split('(' + pattern + ')', text)
    return parts

def pretokenize(parts: list[str], special_tokens: list[str], special_token_to_id: dict[str, int], drop_special_token: bool = False) -> List[List[int]]:
    """
    Convert text parts into pretokens (list of list of ints)
    Special tokens are mapped to their vocab ID directly.
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokens: List[List[int]] = []

    for part in parts:
        if part in special_tokens:
            if not drop_special_token:
                # 直接用特殊 token 的 ID
                pretokens.append([special_token_to_id[part]])
        else:
            str_tokens = re.findall(PAT, part)
            for s in str_tokens:
                pretokens.append([b for b in s.encode('utf-8')])

    return pretokens

def count_pairs(pretokens: List[List[int]]) -> Tuple[dict[Tuple[bytes, bytes], int], dict[Tuple[bytes, bytes], set]]:
    counts = defaultdict(int)
    index_dict = defaultdict(set)  # Store pretoken location for each pair

    for j, pretoken in enumerate(pretokens):
        for index1, index2 in zip(pretoken, pretoken[1:]):
            counts[index1, index2] += 1
            index_dict[index1, index2].add(j)
    return counts, index_dict

def update_pretokens(
    counts: Dict[Tuple[int, int], int],
    index_dict: Dict[Tuple[int, int], Set[int]],
    pretokens: List[List[int]],
    index1: int,
    index2: int,
    new_index: int
):
    """
    用 new_index 替换所有 (index1,index2) 对，并同步更新 counts 和 index_dict。
    """
    pair = (index1, index2)
    if pair not in index_dict:
        return pretokens

    affected_sequences = list(index_dict[pair])

    for seq_idx in affected_sequences:
        if seq_idx >= len(pretokens):
            continue

        seq = pretokens[seq_idx]

        # ---- 第一步：先移除该序列所有 pair 的计数和索引 ----
        for i in range(len(seq) - 1):
            p = (seq[i], seq[i + 1])
            counts[p] = counts.get(p, 0) - 1
            if counts[p] <= 0:
                counts.pop(p, None)
                index_dict.pop(p, None)
            else:
                index_dict[p].discard(seq_idx)

        # ---- 第二步：在当前序列中执行合并 ----
        i = 0
        new_seq = []
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == index1 and seq[i + 1] == index2:
                new_seq.append(new_index)
                i += 2  # 跳过下一个
            else:
                new_seq.append(seq[i])
                i += 1
        pretokens[seq_idx] = new_seq

        # ---- 第三步：重新添加该序列的新 pair 到 counts 和 index_dict ----
        for i in range(len(new_seq) - 1):
            p = (new_seq[i], new_seq[i + 1])
            counts[p] = counts.get(p, 0) + 1
            index_dict.setdefault(p, set()).add(seq_idx)

    # ---- 删除被合并 pair 的记录 ----
    counts.pop(pair, None)
    index_dict.pop(pair, None)

    return pretokens

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    special_tokens = special_tokens or []
    num_merges = max(vocab_size - len(special_tokens) - 256, 0)

    # Initialize vocab
    vocab = {x:bytes([x]) for x in range(0,256)}
    special_token_to_id = {}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")
        special_token_to_id[token] = 256 + i
    merges: List[Tuple[bytes, bytes]] = []

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    parts = split_by_special_tokens(content, special_tokens)

    pretokens: List[List[int]] = []
    pretokens = pretokenize(parts, special_tokens, special_token_to_id, drop_special_token=False)

    # Merging
    counts = defaultdict(int)
    index_dict = defaultdict(set)  # Store pretoken location for each pair
    
    counts, index_dict = count_pairs(pretokens)
    for i in range(num_merges):
        max_pair = max(
            counts.items(),
            key=lambda x: (
                x[1],  # 出现次数
                vocab[x[0][0]].decode("utf-8", errors="ignore"),
                vocab[x[0][1]].decode("utf-8", errors="ignore")
            )
        )[0]

        index1, index2 = max_pair

        new_index = 256 + len(special_tokens) + i

        vocab[new_index] = vocab[index1] + vocab[index2]
        merges.append((vocab[index1], vocab[index2]))
        pretokens = update_pretokens(counts, index_dict, pretokens, index1, index2, new_index)

    return (vocab, merges)
