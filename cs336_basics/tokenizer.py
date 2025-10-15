
from collections.abc import Iterable


class Tokenizer:

    def __init__(self, vocab : dict[int, bytes], merges : list[tuple[bytes, bytes]], special_tokens : list[str] | None = None):
        self.vocav = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def from_files(cls, vocab_filepath : str, merges_filepath : str, special_tokens : list[str] | None = None):
        pass

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[int]) -> Iterable[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass
