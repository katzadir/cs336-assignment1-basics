from collections.abc import Iterable
import pickle
import regex as re


class Tokenizer:

    def __init__(self, vocab : dict[int, bytes], merges : list[tuple[bytes, bytes]], special_tokens : list[str] | None = None):
        """

        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.

        
        """
        self.vocav = vocab
        self.merges = merges
        self.special_tokens = special_tokens

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
    
    def pre_tokenize(input_path, special_tokens):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = {}
    
        try:
            with open(input_path, "r", encoding="utf8") as file:
                text = file.read()
        
                split_pat = "(" + "|".join(re.escape(t) for t in special_tokens) + ")"
                parts = re.split(split_pat, text)
                print("parts| ", len(parts))
                for part in parts:
                    if not part:
                        continue
                    if part in special_tokens:
                        continue
                    for pretoken in re.finditer(PAT, part):
                        token_bytes = pretoken.group(0).encode("utf-8")
                        key = tuple(token_bytes[i:i+1] for i in range(len(token_bytes)))
                        pre_tokens[key] = pre_tokens.get(key, 0) + 1
                print("pre_tokens| ", len(pre_tokens))

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            pass
        return pre_tokens

    def encode(self, text: str) -> list[int]:
        # pre-tokenization
        # apply merges
        # special tokens
        pass

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
        pass
