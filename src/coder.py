import struct
from functools import cached_property, lru_cache
from typing import Iterable

from transformers import AutoTokenizer

from src.knowledge import Descriptive, Embedding, KnowledgeCoder


class AdvancedCoder(KnowledgeCoder):
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
    def tokenizer(self):
        return AutoTokenizer.from_pretrained("google-t5/t5-small")

    def encode_text(self, text: str) -> Iterable:
        """Encode text using gte-small from huggingface."""
        tokens = self.__class__.text_to_tokens(text, self.embedding_size, self.tokenizer)
        size = self.embedding_size // len(tokens)
        encoded_size = self.encode_integer(size)[:4].rjust(4, "0")
        return encoded_size + "".join([self.encode_integer(x)[:size].rjust(size, "0") for x in tokens])

    @lru_cache(maxsize=1000000)
    @staticmethod
    def text_to_tokens(text: str, embedding_size: int, tokenizer: AutoTokenizer) -> Iterable:
        return tokenizer(text).input_ids

    def encode_integer(self, value: int) -> Iterable:
        """Encode integer as a binary vector."""
        return self.__class__._encode_integer(value)

    @lru_cache(maxsize=1000000)
    @staticmethod
    def _encode_integer(value: int) -> Iterable:
        return format(value, "b")

    def encode_float(self, value: float) -> Iterable:
        """Encode float as a binary vector."""
        [d] = struct.unpack(">I", struct.pack(">f", value))
        if self.embedding_size < 32:
            return f"{d:016b}"
        if self.embedding_size < 64:
            return f"{d:032b}"
        return f"{d:064b}"

    def decode_text(self, embedding: Embedding) -> str:
        size = self.decode_integer(Embedding(embedding.data[:4]))
        tokens = []
        for index in range((self.embedding_size - 4) // size):
            tokens.append(self.decode_integer(Embedding(embedding.data[index * size + 4 : (index + 1) * size + 4])))
        return self.tokenizer.decode(tokens)

    def decode_integer(self, embedding: Embedding) -> int:
        return self.__class__._decode_integer(embedding.data)

    @lru_cache(maxsize=1000000)
    @staticmethod
    def _decode_integer(vector: tuple[float, ...]) -> int:
        binary_str = "".join(format(int(x), "b") for x in vector)
        return int(binary_str, 2)

    def decode_float(self, embedding: Embedding) -> float:
        binary_str = "".join(format(max(0, min(1, round(x))), "b") for x in embedding.data)
        if self.embedding_size < 32:
            binary_str = binary_str.zfill(16)[:16]
        elif self.embedding_size < 64:
            binary_str = binary_str.zfill(32)[:32]
        else:
            binary_str = binary_str.zfill(64)[:64]
            return struct.unpack(">d", int(binary_str, 2).to_bytes(8, byteorder="big"))[0]
        packed_value = struct.pack(">I", int(binary_str, 2))
        return struct.unpack(">f", packed_value)[0]
