import math
from functools import cached_property
from typing import Iterable

from src.coder import AdvancedCoder
from src.knowledge import Embedding


class LookupCoder(AdvancedCoder):
    @cached_property
    def text_map(self) -> dict[str, Embedding]:
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

    def encode_integer(self, value: int, embedding_size: int | None = None) -> Iterable:
        embedding_size = embedding_size if embedding_size is not None else self.embedding_size
        if embedding_size == 1:
            return [value]
        if value == 0:
            return [0.0] * embedding_size
        sign = 1.0 if value > 0 else -1.0
        value = abs(value)
        if value <= embedding_size:
            embedding = [0.0] * embedding_size
            embedding[value - 1] = sign
            return embedding
        embedding = self.encode_integer(int(sign * value / (embedding_size + 1)), embedding_size)
        for index, val in enumerate(embedding):
            embedding[index] = val * 0.5
        if value % (embedding_size + 1) > 0:
            embedding[value % (embedding_size + 1) - 1] += sign
        return embedding

    def decode_integer(self, embedding: Embedding, embedding_size: int | None = None) -> int:
        embedding_size = embedding_size if embedding_size is not None else self.embedding_size
        vector = list(embedding.data)
        if embedding_size == 1:
            return vector[0]
        value = 0
        sign = 1
        done = True
        for index, val in enumerate(vector):
            if round(val, 5) != 0:
                done = False
            if abs(val) >= 1:
                sign = 1 if val > 0 else -1
                value += sign * (index + 1)
                vector[index] -= sign
            vector[index] = 2 * vector[index]
        if not done:
            value += self.decode_integer(Embedding(vector), embedding_size) * (embedding_size + 1)
        return value

    def encode_float(self, value: float) -> Iterable:
        if self.embedding_size == 1:
            return [value]
        significand, exponent = math.frexp(value)
        encoded1 = self.encode_integer(int(1000 * significand), self.embedding_size // 2)
        encoded2 = self.encode_integer(int(exponent), self.embedding_size - self.embedding_size // 2)
        return encoded1 + encoded2

    def decode_float(self, embedding: Embedding) -> float:
        vector = embedding.data
        if self.embedding_size == 1:
            return vector[0]
        encoded1 = vector[: self.embedding_size // 2]
        encoded2 = vector[self.embedding_size // 2 :]
        significand = self.decode_integer(Embedding(encoded1), self.embedding_size // 2) / 1000
        exponent = self.decode_integer(Embedding(encoded2), self.embedding_size - self.embedding_size // 2)
        return significand * 2**exponent
