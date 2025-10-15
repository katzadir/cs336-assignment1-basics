
from collections.abc import Iterable


class Tokenizer:

    def __init__(self, vocab : dict[int, bytes], merges : list[tuple[bytes, bytes]], special_tokens : list[str] | None = None):
        """

        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.

        
        """
        self.vocav = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def from_files(cls, vocab_filepath : str, merges_filepath : str, special_tokens : list[str] | None = None):
        """
        Class methos that constructs and return a Tokenizer from a serialized vocabulary
        and a list of merges (in the format created by [train_bpe.py] and (optionally) a
        list of special tokens.)
        """
        pass

    def encode(self, text: str) -> list[int]:
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
