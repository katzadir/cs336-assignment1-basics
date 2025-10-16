from collections.abc import Iterable
import json
import pickle
import regex as re
import os
import tiktoken


class Tokenizer:

    def __init__(self, vocab : dict[int, bytes], merges : list[tuple[bytes, bytes]], special_tokens : list[str] | None = None):
        """

        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.

        
        """
        self.vocav = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # adding special_token to our vocabulary if not exists already
        if special_tokens is not None:
            for special_token in special_tokens:
                if special_token not in self.vocav:
                    self.vocav[len(self.vocav)-1] = special_token.encode("utf-8")

        # inverse vocabulary lookup 
        self.inv_vocab = {v : k for k, v in vocab.items()}

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

        parts = [text]
        if special_tokens is not None:
            split_pat = "(" + "|".join(re.escape(t) for t in special_tokens) + ")"
            parts = re.split(split_pat, text)

        for part in parts:
            if not part:
                continue
            if special_tokens is not None and part in special_tokens:
                pre_tokens.append([part.encode("utf-8")])
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

        pre_tokens = [item for sublist in pre_tokens for item in sublist]

        return pre_tokens




    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Given an iterable of strings(e.g. a Python file handle), 
        return a generator that laizily yields token IDs.

        This is required for memoery-efficient tokenization of large files that we cannot directly load
        into memory.
        """
        while iter in iterable:
            yield self.encode(iter)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequance of token IDs into text.
        """

        str = [self.vocav.get(id, '\uFFFD') for id in ids]
        str = b"".join(str)
        return str.decode("utf-8","replace")
        #return b'\xf0\x9f\x99\x83'.decode("utf-8")
    

    @classmethod
    def get_tokenizer(

        special_tokens: None = None,
    ):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        vocab_path = "tests/fixtures/gpt2_vocab.json"
        merges_path = "tests/fixtures/gpt2_merges.txt"
        with open(vocab_path) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_path, encoding="utf-8") as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        #if special_tokens is not None:
        #    for special_token in special_tokens:
        #        byte_encoded_special_token = special_token.encode("utf-8")
        #        if byte_encoded_special_token not in set(vocab.values()):
        #            vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return Tokenizer(vocab, merges)

def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(merges_filepath="merges_v1.pkl", vocab_filepath="vocab_v1.pkl", special_tokens=["<|endoftext|>"])

    # print("the <|endoftext|> are")
    # print(tokenizer.encode("the <|endoftext|> are"))
    # print(tokenizer.decode(tokenizer.encode("the <|endoftext|> are")))
    merges_filepath="merges_v1.pkl"
    vocab_filepath="vocab_v1.pkl"
    
    VOCAB_PATH = "tests/fixtures/gpt2_vocab.json"
    MERGES_PATH = "tests/fixtures/gpt2_merges.txt"
    #tokenizer = Tokenizer.get_tokenizer()

    tokenizer = Tokenizer.from_files(merges_filepath=merges_filepath, vocab_filepath=vocab_filepath, special_tokens=["<|endoftext|>"])
    #orig_str = "Hello, how are you?"
    orig_str = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    #orig_str = "🙃"
    print("orig: ",orig_str)
    #print("orig-encoded: ",orig_str.encode("utf-8"))
    #print(type(tokenizer.encode(orig_str)))
    print("encoded: ",tokenizer.encode(orig_str), " len: ",len(tokenizer.encode(orig_str)))
    #print(type(tokenizer.encode(orig_str)[0]))
    print("deco: ",tokenizer.decode(tokenizer.encode(orig_str)))

    reference_tokenizer = tiktoken.get_encoding("gpt2")
    #print("ref encoding:", reference_tokenizer.encode(orig_str))

    # print(tokenizer.decode(tokenizer.encode("")))
    # print(tokenizer.decode(tokenizer.encode("a")))
    # print(tokenizer.encode("a"))