from functools import cached_property, lru_cache
from typing import Iterable

from transformers import AutoTokenizer

from src.coder import FloatCoder
from src.knowledge import Embedding


@lru_cache(maxsize=1000000)
def encode_text(text: str, embedding_size: int, tokenizer: AutoTokenizer) -> Iterable:
    tokens = tokenizer(text).input_ids
    return encode_integer_list(tuple(tokens), embedding_size)


@lru_cache(maxsize=1000000)
def encode_integer_list(values: tuple[int], embedding_size: int) -> Iterable:
    size = embedding_size // len(values)
    encoded_size = encode_integer(size)[:4].rjust(4, "0")
    return encoded_size + "".join([encode_integer(x)[:size].rjust(size, "0") for x in values])


@lru_cache(maxsize=1000000)
def encode_integer(value: int) -> Iterable:
    return format(value, "b")


@lru_cache(maxsize=1000000)
def decode_integer(vector: tuple[float, ...]) -> int:
    binary_str = "".join(format(int(x), "b") for x in vector)
    return int(binary_str, 2)


class TextCoder(FloatCoder):
    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained("google-t5/t5-small")

    def encode_text(self, text: str) -> Iterable:
        return encode_text(text, self.embedding_size, self.tokenizer)

    def decode_text(self, embedding: Embedding) -> str:
        size = decode_integer(Embedding(embedding.data[:4]))
        tokens = []
        for index in range((self.embedding_size - 4) // size):
            tokens.append(decode_integer(Embedding(embedding.data[index * size + 4 : (index + 1) * size + 4])))
        return self.tokenizer.decode(tokens)
