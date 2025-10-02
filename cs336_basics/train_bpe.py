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

        
    # compute byte-pair stat 
    pairs_stat = dict()
    for pre_token in pre_tokens:
        # count byte-pairs per pre-token
        for x,y in zip(pre_token[:-1],pre_token[1:]):
            pairs_stat[(x,y)] = pairs_stat.get((x,y), 0) + pre_tokens[pre_token]

    # bpe merge
    merges = list()
    while len(vocab) < vocab_size:
        pre_tokens, pairs_stat, vocab, merges = bpe_merge(pre_tokens, pairs_stat, vocab, merges)

    return vocab, merges

def bpe_merge(pre_tokens, pairs_stat, vocab, merges):
    
    # high-freq pair
    merge_cand = max(pairs_stat, key=lambda k: (pairs_stat[k], k))

    # merge update
    merges.append(merge_cand)

    # update vocab
    merged_bp = merge_cand[0] + merge_cand[1]
    token_idx = len(vocab)
    vocab[token_idx] = merged_bp

    # update pre_tokens + byte-pair stat
    pre_tokens_new = {}
    
    for pre_token, freq in pre_tokens.items():
        #key = pre_token
        idx = 0
        while idx < len(pre_token):
            if pre_token[idx:idx+2] == merge_cand:

                # update stat of new token pairs
                if pre_token[:idx]:
                    key_l = (pre_token[idx-1], merged_bp)
                    pairs_stat[key_l] = pairs_stat.get(key_l, 0) + freq
                    pairs_stat[(pre_token[idx-1], pre_token[idx])] -= freq
                
                if pre_token[idx+2:]:
                    key_r = (merged_bp, pre_token[idx + 2])
                    pairs_stat[key_r] = pairs_stat.get(key_r, 0) + freq
                    pairs_stat[(pre_token[idx+1], pre_token[idx+2])] -= freq
                
                pairs_stat[merge_cand] -= freq

                # create merged pretoken
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




