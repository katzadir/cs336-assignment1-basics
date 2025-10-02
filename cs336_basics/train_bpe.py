import regex as re
from tqdm import tqdm

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):# -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # vocab init
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256+i] = token.encode("utf-8")

    # Pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens = {}
    
    try:
        with open(input_path, "r", encoding="utf8") as file:
            text = file.read()
        
        split_pat = "(" + "|".join(re.escape(t) for t in special_tokens) + ")"
        parts = re.split(split_pat, text)

        pre_tokens = {}  # dict[bytes, int] or dict[str, int], your choice
        for part in parts:
            if not part:
                continue
            if part in special_tokens:
                continue
            for pretoken in re.finditer(PAT, part):
                token_bytes = pretoken.group(0).encode("utf-8")
                key = tuple(token_bytes[i:i+1] for i in range(len(token_bytes)))
                #key = tuple(char.encode('utf-8') for char in pretoken.group())
                pre_tokens[key] = pre_tokens.get(key, 0) + 1

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        file.close()

    # bpe merge
    merges = list()
    while len(vocab) < vocab_size:
        pairs_stat = dict()
        pre_tokens, pairs_stat, vocab, merges = bpe_merge(pre_tokens, pairs_stat, vocab, merges)

    return vocab, merges

def bpe_merge(pre_tokens, pairs_stat, vocab, merges):
    
    
    # compute byte-pair stat 
    for pre_token in pre_tokens:
        # count byte-pairs per pre-token
        for x,y in zip(pre_token[:-1],pre_token[1:]):
            pairs_stat[(x,y)] = pairs_stat.get((x,y), 0) + pre_tokens[pre_token]

    merge_cand = max(pairs_stat, key=lambda k: (pairs_stat[k], k))

    # merge update
    merges.append(merge_cand)

    # update vocab
    merged_bp = merge_cand[0] + merge_cand[1]
    vocab[len(vocab)] = merged_bp

    # update pre_tokens + byte-pair stat
    pre_tokens_new = {}
    
    for pre_token, freq in pre_tokens.items():
        #key = pre_token
        idx = 0
        while idx < len(pre_token):
            if pre_token[idx:idx+2] == merge_cand:
                lst = list(pre_token)
                lst[idx] = merged_bp
                lst.pop(idx+1)
                pre_token = tuple(lst) 
            idx += 1
        pre_tokens_new[pre_token] = freq
    pre_tokens = pre_tokens_new    

    return pre_tokens, pairs_stat, vocab, merges


if __name__ == "__main__":
    input_path="data/TinyStoriesV2-GPT4-valid.txt"
    FIXTURES_PATH =  "tests/fixtures"
    input_path = FIXTURES_PATH + "/corpus.en"
    input_path = FIXTURES_PATH + "/tinystories_sample_5M.txt"

    special_tokens = [("<|endoftext|>"),]

    vocab, merges = train_bpe(input_path, vocab_size=1000, special_tokens=special_tokens)

    # Open the file in write mode ('w') and use json.dump() to write the dictionary
    # vocab_swapped = {
    #     k : v for v, k in vocab.items()
    # }

    # vocab_serializable = {k: v for v, k in vocab.items()}
    # with open("./vocab.json", 'w', encoding="utf-8") as f:
    #     json.dump(vocab_swapped, f, ensure_ascii=False, indent=4)
    
    # with open("./merges.txt", 'w') as f:
    #     for x, y in merges:
    #         f.write(f"{x.decode('utf-8')} {y.decode('utf-8')}\n")

    print("vocab size: ", len(vocab))
    for x,y in vocab.items():
        print(x , "-->",y)




