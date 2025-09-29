import regex as re
from tqdm import tqdm

def train_bpe(input_path: str, PAT: str, vocab_size: int, special_tokens: list[str], log=False):# -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # iterate over the file, split according to PAT, then convert to UTF-8
    # count indices of byte-pairs in a dictionary
    # sort, select the biggest and add to the vocabulary
    vocab = dict() #dict((x, chr(x)) for x in range(5))
    idx = 256 # len(vocab) + 1
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # initial conversion to utf-8
    token_stream = list()
    try:
        it = 0;
        with open(input_path, "r", encoding="utf8") as file:
            for line in tqdm(file,desc="processing file"):
                #print(line)
                if it == -1:
                    break
                it += 1
                for r in re.finditer(PAT, line):
                    res = r.group()
                    token_stream.append(list(res.encode("utf-8")))
                    # if log == True:
                    #     print(res, "==>", list(res.encode("utf-8")))
                
        file.close()

    except FileNotFoundError:
        print("Error: The file 'sample.txt' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        file.close()

    # extract pyte pairs for all words in the document
    num_lines = len(token_stream)
    print(" num lines: ", num_lines)
    pairs_stat, token_stream = gather_pair_stat(token_stream, PAT, vocab, num_lines)

    # update vocabulary
    while True:
        if len(vocab) > vocab_size:# or len(pairs_stat) == 0:
            break
        # Sort by value in descending order
        freq_pair = tuple(max(pairs_stat, key=pairs_stat.get))
        
        if freq_pair in vocab:
            pairs_stat.pop(freq_pair)
        else:
            if (log == True):
                print(freq_pair, "##-->", pairs_stat[freq_pair], "idx:", idx)
            vocab[freq_pair] = idx
            idx += 1
            pairs_stat, token_stream = gather_pair_stat(token_stream, PAT, vocab, num_lines)
            





        
        # pairs_stat = dict(sorted(pairs_stat.items(), key=lambda item: item[1], reverse=True))
        # print(pairs_stat[1:10])


    return vocab, pairs_stat, token_stream

# encode `pairs` according to `vocab``
def encode_pairs(pairs, vocab, token_stream, log=False):
    
    new_pairs = pairs.copy()
    some_change = True
    if (len(pairs) > 1):
        while some_change is True:
            idx = 0
            some_change = False
            while idx < (len(new_pairs)-1):
                x = new_pairs[idx]
                # if idx >= len(pairs)-1:
                #     new_pairs.append(x)
                #     break
                y = new_pairs[idx+1]
                if (x, y) in vocab:
                    if (log is True):
                        print("replaced: (", x,",", y,") ==>", vocab[(x,y)])
                    new_pairs[idx] = vocab[(x,y)]
                    new_pairs.pop(idx+1)

                    # # op1: x is in idx 0
                    # if (x,y) in token_stream:
                    #     token_stream[(x,y)] += 1
                    # else:
                    #     token_stream[(x,y)] = 1
                    token_stream  # can I locally update the stat around the new merge?
                    idx +=1
                    some_change = True
                else:
                    idx +=1
            
    return new_pairs


def gather_pair_stat(token_stream, PAT, vocab, num_lines):
    num_iter = -1
    pairs_stat = {}
    for index, line in tqdm(enumerate(token_stream), desc="processing file", total=num_lines):
        if num_iter ==0:
            break
        num_iter -= 1
        pairs_enc = encode_pairs(line, vocab, token_stream)
        # if len(line) != len(pairs_enc):
        #     print(line, "===>", pairs_enc)
        token_stream[index] = pairs_enc
        if (len(pairs_enc) >1):
            for x,y in zip(pairs_enc[:-1],pairs_enc[1:]):
                if (x,y) not in pairs_stat:
                    pairs_stat[(x,y)] = 1
                else:
                    pairs_stat[(x,y)] += 1
    return pairs_stat, token_stream


import os




current_working_directory = os.getcwd()
print(current_working_directory)

input_path="data/TinyStoriesV2-GPT4-valid.txt"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
vocab, pairs, token_stream = train_bpe(input_path, PAT=PAT, vocab_size=50, special_tokens=("<|endoftext|>"), log=True)
print("vocab size: ", len(vocab))
for x,y in vocab.items():
    print(x , "-->",y)
print("===================================")
it = 0
with open(input_path, "r", encoding="utf8") as file:
            for line in tqdm(file,desc="processing file"):
                print("line:", line)
                if it == 2:
                    break
                it += 1
                for r in re.finditer(PAT, line):
                    res = r.group()
                    utf8_enc = list(res.encode("utf-8"))
                    encoded = encode_pairs(utf8_enc, vocab, True)
                    if len(utf8_enc)!=len(encoded):
                        print(res, "==> orig:", utf8_enc, "final: ", encoded)
                    else:
                        print(res)
                
file.close()

#print(token_stream)
print(len(vocab))




