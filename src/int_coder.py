from typing import Iterable

from src.coder import AdvancedCoder
from src.knowledge import Embedding


class IntCoder(AdvancedCoder):
    def encode_integer(self, value: int) -> Iterable:
        return format(value, "b")

    def decode_integer(self, embedding: Embedding) -> int:
        binary_str = "".join(format(max(0, min(1, round(x))), "b") for x in embedding.data)
        return int(binary_str, 2)

    def encode_float(self, value: float) -> Iterable:
        return self.encode_integer(round(value * 1000))

    def decode_float(self, embedding: Embedding) -> float:
        return self.decode_integer(embedding) / 1000
