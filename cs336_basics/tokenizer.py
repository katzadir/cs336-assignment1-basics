
class Tokenizer:

    def __init__(self, vocab : dict[int, bytes], merges : list[tuple[bytes, bytes]], special_tokens : list[str] | None = None):
        self.vocav = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    
