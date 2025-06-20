import math
from functools import cached_property
from typing import Iterable

import numpy as np

from src.coder import AdvancedCoder
from src.knowledge import Embedding


class LookupCoder(AdvancedCoder):
    @cached_property
    def text_map(self) -> tuple[dict[str, Embedding], dict[Embedding, str], list[int]]:
        return {}, {}, [0]

    def encode_text(self, text: str) -> Iterable:
        text_map, text_map_rev, index = self.text_map
        if text not in text_map:
            text_map[text] = self.encode_integer(index[0])
            text_map_rev[tuple(text_map[text])] = text
            index[0] += 1
        return text_map[text]

    def decode_text(self, embedding: Embedding) -> str:
        text_map, text_map_rev, index = self.text_map
        if embedding.data in text_map_rev:
            return text_map_rev[embedding.data]
        return ""

    def encode_integer(self, value: int) -> Iterable:
        return self.encode_float(value)

    def decode_integer(self, embedding: Embedding) -> int:
        return self.decode_float(embedding)

    def encode_small_integer(self, value: int, embedding_size: int | None = None) -> Iterable:
        sign = int(np.sign(value))
        value = abs(value)
        result = format(value, "b")
        if embedding_size:
            result = result.zfill(embedding_size)[-embedding_size:]
        return [sign * int(x) for x in result]

    def decode_small_integer(self, embedding: Embedding) -> int:
        binary_str = "".join(str(max(0, min(1, round(abs(x))))) for x in embedding.data)
        sign = int(np.sign(sum(np.sign(int(x)) for x in embedding.data)))
        return sign * int(binary_str, 2)

    def encode_binary(self, value: float, embedding_size: int | None = None) -> Iterable:
        """Encodes numbers from [-1, 1]"""
        embedding_size = embedding_size if embedding_size else self.embedding_size
        x = value
        result = []
        for index in range(embedding_size - 1):
            x = (x - int(x)) * 2
            result.append(int(x))
        result.append((x - int(x)) * 2)
        return result

    def decode_binary(self, embedding: Embedding) -> int:
        embedding_size = len(embedding.data)
        return sum(embedding.data[index] * 0.5 ** (index + 1) for index in range(embedding_size))

    def encode_float(self, value: float) -> Iterable:
        if self.embedding_size == 1:
            return [value]
        significand, exponent = math.frexp(value)
        significant_len = max(self.embedding_size // 2, self.embedding_size - 4)
        encoded1 = self.encode_binary(significand, significant_len)
        encoded2 = self.encode_small_integer(exponent, self.embedding_size - significant_len)
        return encoded1 + encoded2

    def decode_float(self, embedding: Embedding) -> float:
        vector = embedding.data
        if self.embedding_size == 1:
            return vector[0]
        significant_len = max(self.embedding_size // 2, self.embedding_size - 4)
        encoded1 = vector[:significant_len]
        encoded2 = vector[significant_len:]
        significand = self.decode_binary(Embedding(encoded1))
        exponent = self.decode_small_integer(Embedding(encoded2))
        return significand * 2**exponent
