import os
import regex as re
from typing import BinaryIO, Tuple, Set, List, Dict, Iterable, Iterator
from collections import defaultdict
import json

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

def pretokenize(
    parts: List[str], 
    special_tokens: List[str], 
    special_token_to_id: Dict[str, int], 
    drop_special_token: bool = False) -> Dict[Tuple[int, ...], int]:
    """
    Convert text parts into pretokens and count their frequencies.
    Special tokens are mapped to their vocab ID directly.
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokens_freq: Dict[Tuple[int, ...], int] = {}
    
    for part in parts:
        if part in special_tokens:
            if not drop_special_token:
                # 创建包含特殊token ID的元组
                token_tuple = (special_token_to_id[part],)
                # 更新频率计数
                pretokens_freq[token_tuple] = pretokens_freq.get(token_tuple, 0) + 1
        else:
            # 使用正则表达式分割文本
            str_tokens = re.findall(PAT, part)
            for s in str_tokens:
                # 将字符串转换为字节，然后转换为整数元组
                byte_tokens = tuple(b for b in s.encode('utf-8'))
                # 更新频率计数
                pretokens_freq[byte_tokens] = pretokens_freq.get(byte_tokens, 0) + 1
    
    return pretokens_freq

def count_pairs(pretokens_freq: Dict[Tuple[int, ...], int]) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], Set[Tuple[int, ...]]]]:
    counts = defaultdict(int)
    index_dict = defaultdict(set)
    
    for pretoken_key, frequency in pretokens_freq.items():
        # 直接遍历整数元组，统计相邻整数对
        for i in range(len(pretoken_key) - 1):
            pair = (pretoken_key[i], pretoken_key[i + 1])
            counts[pair] += frequency
            index_dict[pair].add(pretoken_key)
    
    return dict(counts), dict(index_dict)

def update_pretokens(
    counts: Dict[Tuple[int, int], int],
    index_dict: Dict[Tuple[int, int], Set[Tuple[int, ...]]],
    pretokens_freq: Dict[Tuple[int, ...], int],
    index1: int,
    index2: int,
    new_index: int
) -> Dict[Tuple[int, ...], int]:
    """
    用 new_index 替换所有 (index1,index2) 对，并同步更新 counts 和 index_dict。
    返回更新后的 pretokens_freq 字典。
    """
    # 将整数索引转换为字节对
    pair = (index1, index2)
    
    if pair not in index_dict:
        return pretokens_freq

    # 获取包含该字节对的所有pretoken键
    affected_pretoken_keys = list(index_dict[pair])
    
    # 创建更新后的pretokens_freq字典
    updated_pretokens_freq = pretokens_freq.copy()
    
    for pretoken_key in affected_pretoken_keys:
        if pretoken_key not in updated_pretokens_freq:
            continue
            
        frequency = updated_pretokens_freq[pretoken_key]
        
        # ---- 第一步：先移除该pretoken所有pair的计数和索引 ----
        pretoken = tuple(b for b in pretoken_key)
        for i in range(len(pretoken) - 1):
            p = (pretoken[i], pretoken[i + 1])
            if p in counts:
                counts[p] = counts.get(p, 0) - frequency
                if counts[p] <= 0:
                    counts.pop(p, None)
                    index_dict.pop(p, None)
                else:
                    index_dict[p].discard(pretoken_key)

        # ---- 第二步：在当前pretoken中执行合并 ----
        int_pretoken = list(pretoken_key)
        i = 0
        new_pretoken = []
        while i < len(int_pretoken):
            if i < len(int_pretoken) - 1 and int_pretoken[i] == index1 and int_pretoken[i + 1] == index2:
                new_pretoken.append(new_index)
                i += 2  # 跳过下一个
            else:
                new_pretoken.append(int_pretoken[i])
                i += 1
        
        new_pretoken_key = tuple(new_pretoken)
        
        # 更新pretokens_freq
        updated_pretokens_freq.pop(pretoken_key, None)
        if new_pretoken_key:  # 避免空序列
            updated_pretokens_freq[new_pretoken_key] = updated_pretokens_freq.get(new_pretoken_key, 0) + frequency

        # ---- 第三步：重新添加新pretoken的pair到counts和index_dict ----
        if new_pretoken_key:
            new_pretoken = tuple(b for b in new_pretoken_key)
            for i in range(len(new_pretoken) - 1):
                p = (new_pretoken[i], new_pretoken[i + 1])
                counts[p] = counts.get(p, 0) + frequency
                index_dict.setdefault(p, set()).add(new_pretoken_key)

    # ---- 删除被合并pair的记录 ----
    counts.pop(pair, None)
    index_dict.pop(pair, None)

    return updated_pretokens_freq

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

    # pretokens: List[List[int]] = []
    pretokens: Dict[Tuple[int, ...], int] = []
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

class BPETokenizer:
    def __init__(self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None):

        self.id_to_token = dict(vocab)
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.special_tokens = set(special_tokens or [])
        self.merge_dict_tuple = {(self.token_to_id[a], self.token_to_id[b]): i for i, (a, b) in enumerate(merges)}
        self.merges = merges

    def pretokenize(self, parts: list[str], drop_special_token: bool = False) -> List[List[int]]:
        """
        Convert text parts into pretokens (list of list of ints)
        Special tokens are mapped to their vocab ID directly.
        """
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pretokens: List[List[int]] = []

        for part in parts:
            if part in self.special_tokens:
                if not drop_special_token:
                    # 直接用特殊 token 的 ID
                    pretokens.append([self.token_to_id[part.encode("utf-8")]])
            else:
                str_tokens = re.findall(PAT, part)
                for s in str_tokens:
                    pretokens.append([self.token_to_id[bytes([b])] for b in s.encode("utf-8")])

        return pretokens
    
    def atom_token_to_id(self, atom: list[int]) -> list[int]:
        """
        将 atom (字节或字符的 ID 列表) 通过 BPE 规则不断合并成最终 token ID 列表
        """
        if len(atom) < 2: 
            return atom
        while True:
            merge_idx = None
            min_rank = float('inf')
            new_id = None

            # 遍历找最小 rank 的 pair
            for idx in range(len(atom) - 1):
                pair = (atom[idx], atom[idx + 1])
                if pair in self.merge_dict_tuple:
                    rank = self.merge_dict_tuple[pair]
                    if rank < min_rank:
                        min_rank = rank
                        new_id = self.token_to_id[self.id_to_token[pair[0]] + self.id_to_token[pair[1]]]
                        merge_idx = idx

            # 如果找不到可合并的 pair，则退出
            if merge_idx is None:
                break

            # 执行合并：用 new_id 替换掉 atom[merge_idx], atom[merge_idx+1]
            atom = atom[:merge_idx] + [new_id] + atom[merge_idx+2:]

        return atom

    def encode(self, text: str) -> list[int]:
        split_text = split_by_special_tokens(text, self.special_tokens)
        token_id_list: list[list[int]] = self.pretokenize(split_text)
        
        token_ids: list[int] = []
        for atom in token_id_list:
            # atom 是 list[int]，BPE 合并
            token_ids.extend(self.atom_token_to_id(atom))

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        对超出 vocab 范围的 token 使用 Unicode Replacement Character (U+FFFD)。
        """
        tokens = bytes()
        vocab_size = len(self.id_to_token)
        replacement_char = "\uFFFD"  # Unicode replacement character

        for token_id in token_ids:
            if token_id < vocab_size:
                token = self.id_to_token[token_id]  # bytes
            else:
                # 对超出范围的 token 使用 replacement char
                token = replacement_char.encode('utf-8')

            tokens += token

        decoded_text = tokens.decode('utf-8', errors='replace')
        return decoded_text
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs. 
        This is required for memory-eﬀicient tokenization of large files 
        that we cannot directly load into memory.
        """
        for line in iterable:
            for idx in self.encode(line):
                yield idx
    
    def save(self, file_path: str) -> None:
        """
        Save the tokenizer's vocabulary, merges, and special tokens to a JSON file.
        """
        # 保存 vocab: int -> str
        vocab_serializable = {str(k): v.decode('utf-8', errors='replace') for k, v in self.id_to_token.items()}
        
        # 保存 merges: tuple[bytes, bytes] -> tuple[str, str]
        merges_serializable = [(a.decode('utf-8', errors='replace'), b.decode('utf-8', errors='replace')) 
                               for a, b in self.merges]
        
        data = {
            'vocab': vocab_serializable,
            'merges': merges_serializable,
            'special_tokens': list(self.special_tokens)
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path: str) -> 'BPETokenizer':
        """
        Load a tokenizer from a JSON file.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 将 vocab 字符串转回 bytes
        vocab = {int(k): v.encode('utf-8') for k, v in data['vocab'].items()}
        
        # 将 merges 字符串转回 bytes
        merges = [(a.encode('utf-8'), b.encode('utf-8')) for a, b in data['merges']]
        
        special_tokens = data.get('special_tokens', [])
        
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
