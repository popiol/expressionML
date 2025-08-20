import math
from functools import cached_property
from typing import Iterable

import numpy as np

from src.knowledge import Descriptive, Embedding, KnowledgeCoder


class FloatCoder(KnowledgeCoder):
    def encode(self, value: Descriptive) -> Embedding:
        if isinstance(value, str):
            vector = self.encode_text(value)
        elif isinstance(value, int):
            vector = self.encode_integer(value)
        elif isinstance(value, float):
            vector = self.encode_float(value)
        return self.to_embedding(vector)

    def to_embedding(self, vector: Iterable) -> Embedding:
        embedding = [float(x) for x in vector]
        embedding = [0.0] * (self.embedding_size - len(embedding)) + embedding
        embedding = embedding[: self.embedding_size]
        return Embedding(tuple(embedding))

    def decode(self, embedding: Embedding, output_type: type[Descriptive]) -> Descriptive:
        if output_type is str:
            return self.decode_text(embedding)
        elif output_type is int:
            return self.decode_integer(embedding)
        elif output_type is float:
            return self.decode_float(embedding)
        raise ValueError(f"Unsupported type for decoding: {output_type}")

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
        return round(self.decode_float(embedding))

    def encode_small_integer(self, value: int, embedding_size: int | None = None) -> list[float]:
        sign = int(np.sign(value))
        value = abs(value)
        result = format(value, "b")
        if embedding_size:
            if embedding_size > len(result):
                result = result.zfill(embedding_size)
            elif embedding_size < len(result):
                last_str = result[embedding_size - 1 :]
                last = int(last_str, 2)
                result = [
                    int(x) * 2 ** (len(result) - embedding_size)
                    for x in result[: embedding_size - 1]
                ] + [last]
        return [sign * int(x) for x in result]

    def decode_small_integer(self, embedding: Embedding) -> int:
        result = 0
        for val in embedding.data:
            result = result * 2 + val
        return int(result)

    def encode_binary(self, value: float, embedding_size: int | None = None) -> list[float]:
        """Encodes numbers from [-1, 1]"""
        embedding_size = embedding_size if embedding_size else self.embedding_size
        x = value
        result = []
        for index in range(embedding_size - 1):
            x = (x - int(x)) * 2
            result.append(int(x))
        result.append(x - int(x))
        return result

    def decode_binary(self, embedding: Embedding) -> float:
        embedding_size = len(embedding.data)
        last = embedding.data[embedding_size - 1] * 0.5 ** (embedding_size - 1)
        return (
            sum(embedding.data[index] * 0.5 ** (index + 1) for index in range(embedding_size - 1))
            + last
        )

    def encode_float(self, value: float) -> Iterable:
        if self.embedding_size == 1:
            return [value]
        significand, exponent = math.frexp(value)
        significant_len = max(self.embedding_size // 2, self.embedding_size - 4)
        encoded1 = self.encode_binary(significand, significant_len)
        encoded2 = self.encode_small_integer(exponent, self.embedding_size - significant_len)
        return encoded1 + encoded2

    def decode_float(self, embedding: Embedding) -> float:
        significant_len = max(self.embedding_size // 2, self.embedding_size - 4)
        vector: list[float] = [round(x) for x in embedding.data]
        vector[significant_len-1] = embedding.data[significant_len-1]
        if self.embedding_size == 1:
            return vector[0]
        encoded1 = vector[:significant_len]
        encoded2 = vector[significant_len:]
        significand = self.decode_binary(Embedding(encoded1))
        exponent = self.decode_small_integer(Embedding(encoded2))
        return significand * 2**exponent
