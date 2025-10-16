from cs336_basics.train_bpe import train_bpe
import pickle

# args
text_source = 'data/raw/owt_train.txt'
output_vocab_path = 'data/out/owt_vocab.pkl'
output_merge_path = 'data/out/owt_merges.pkl'
special_tokens = ['<|endoftext|>']
vocab_size = 32*(10**3)

# training
vocab, merges = train_bpe(input_path=text_source, vocab_size=vocab_size, special_tokens=special_tokens)

# Serialize and save the vocab and merges
with open(output_vocab_path, "wb") as f:
    pickle.dump(merges, f) 

with open(output_vocab_path, "wb") as f:
    pickle.dump(vocab, f) 