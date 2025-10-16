import numpy as np
import tqdm

from cs336_basics.tokenizer import Tokenizer

owt_conf = {
    'train':'data/raw/owt_train.txt',
    'val':'data/raw/owt_valid.txt',
    'vocab_filepath': 'data/out/owt_vocab.json',
    'merges_filepath': 'data/out/owt_merges.txt',
    'special_tokens': ['<|endoftext|>']
}

tokenizer = Tokenizer.from_files(merges_filepath=owt_conf['merges_filepath'], 
                                 vocab_filepath=owt_conf['vocab_filepath'],
                                 special_tokens=owt_conf['special_tokens'])

for split in ['train', 'val']:
    with open(owt_conf[split]) as f:
        text = f.read()
    encoded = tokenizer.encode(text)

    # save the ids
    total_batches = 1024
    batch_size = len(encoded) // total_batches
    arr = np.memmap(f'data/owt/{split}.bin', dtype=np.uint16, mode='w+', shape=(len(encoded),))
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'Writing {split}.bin'):
        batch = encoded[idx:idx+batch_size]
        arr[idx:idx+batch_size] = batch
        idx += batch_size
arr.flush()