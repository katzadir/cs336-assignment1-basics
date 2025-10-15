from collections.abc import Iterable
import pickle
import regex as re


class Tokenizer:

    def __init__(self, vocab : dict[int, bytes], merges : list[tuple[bytes, bytes]], special_tokens : list[str] | None = None):
        """

        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.

        
        """
        self.vocav = vocab
        # inverse vocabulary lookup 
        self.inv_vocab = {v : k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

        # adding special_token to our vocabulary if not exists already
        if special_tokens is not None:
            for special_token in special_tokens:
                self.vocav[len(self.vocav)] = special_token

    @classmethod
    def from_files(cls, vocab_filepath : str, merges_filepath : str, special_tokens : list[str] | None = None):
        """
        Class methos that constructs and return a Tokenizer from a serialized vocabulary
        and a list of merges (in the format created by [train_bpe.py] and (optionally) a
        list of special tokens.)
        """
        try:
            with open(vocab_filepath, 'rb') as f:
                vocab = pickle.load(f)
            with open(merges_filepath, 'rb') as f:
                merges = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: The file {vocab_filepath} or {merges_filepath} (or both) was not found.")
        except Exception as  e:
            print(f"An error occured during deserialization: {e}")

        return cls(vocab, merges, special_tokens)
    
    def pre_tokenize(self, text, special_tokens):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = list()

        parts = re.split(" ", text)
        if special_tokens is not None:
            split_pat = "(" + "|".join(re.escape(t) for t in special_tokens) + ")"
            parts = re.split(split_pat, text)

        for part in parts:
            if not part:
                continue
            if special_tokens is not None and part in special_tokens:
                continue
            for pretoken in re.finditer(PAT, part):
                token_bytes = pretoken.group(0).encode("utf-8")
                pre_token = tuple(token_bytes[i:i+1] for i in range(len(token_bytes)))
                pre_tokens.append(pre_token)

        return pre_tokens

    def encode(self, text: str) -> list[int]:

        # pre-tokenization
        pre_tokens = self.pre_tokenize(text, self.special_tokens)

        # apply merges in the same order as in merges
        pre_token_id = 0
        while pre_token_id < len(pre_tokens):
            pre_token = pre_tokens[pre_token_id]
            updated_pre_token = list(pre_token)
            if (len(pre_token) > 1):
                for merge in self.merges:
                    idx = 0
                    while idx < len(pre_token) and len(pre_token) > 1:
                        if pre_token[idx:idx+2] == merge:
                            updated_pre_token[idx] = pre_token[idx] + pre_token[idx+1]
                            updated_pre_token.pop(idx+1)
                            pre_token = tuple(updated_pre_token) 
                        
                        idx += 1
                
            # now encode using the vocabulary
            pre_tokens[pre_token_id] = [self.inv_vocab[tok] for tok in pre_token]
            pre_token_id += 1

        # special tokens

        return pre_tokens




    def encode_iterable(self, iterable: Iterable[int]) -> Iterable[int]:
        """
        Given an iterable of strings(e.g. a Python file handle), 
        return a generator that laizily yields token IDs.

        This is required for memoery-efficient tokenization of large files that we cannot directly load
        into memory.
        """
        pass

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequance of token IDs into text.
        """

        str = [self.vocav.get(id[0], '\uFFFD').decode('utf-8',"replace") for id in ids]
        return "".join(str)



if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(merges_filepath="merges_v1.pkl", vocab_filepath="vocab_v1.pkl", special_tokens=["<|endoftext|>"])
    print(tokenizer.encode("the <|endoftext|> are"))
    print(tokenizer.decode(tokenizer.encode("the <|endoftext|> are")))

    tokenizer = Tokenizer.from_files(merges_filepath="merges_v1.pkl", vocab_filepath="vocab_v1.pkl")
    print(tokenizer.decode(tokenizer.encode("the are")))

    print(tokenizer.decode(tokenizer.encode("")))