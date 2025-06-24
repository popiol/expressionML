from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

Descriptive = str | int | float
FLOAT_TYPE = np.float32


@dataclass
class Embedding:
    data: tuple[float, ...]

    @staticmethod
    def from_numpy(array: np.ndarray) -> Embedding:
        return Embedding(tuple(array.flatten().tolist()))

    def to_numpy(self) -> np.ndarray:
        return np.array(self.data, dtype=FLOAT_TYPE)


@dataclass
class KnowledgeCoder:
    embedding_size: int

    def encode(self, value: Descriptive) -> Embedding:
        if isinstance(value, str):
            return Embedding(tuple(ord(c) for c in value.ljust(self.embedding_size)[: self.embedding_size]))
        elif isinstance(value, (int, float)):
            return Embedding(tuple([float(value)] + [0.0] * (self.embedding_size - 1)))

    def decode(self, embedding: Embedding, output_type: type[Descriptive]) -> Descriptive:
        if output_type is str:
            return "".join(chr(int(c)) for c in embedding.data)
        elif output_type is int:
            return int(embedding.data[0])
        elif output_type is float:
            return float(embedding.data[0])
        raise ValueError(f"Unsupported type for decoding: {output_type}")

    def with_embedding_size(self, embedding_size: int):
        return self.__class__(embedding_size)


@dataclass
class AtomicFormat:
    key: Descriptive
    value_type: type[Descriptive]
    encoded_key: Embedding
    encoded_value_length: int


@dataclass
class KnowledgeFormat:
    format: list[AtomicFormat]

    @property
    def size(self) -> int:
        return sum(af.encoded_value_length for af in self.format)

    def __hash__(self):
        return hash(tuple(af.key for af in self.format))


@dataclass
class AtomicKnowledge:
    coder: KnowledgeCoder

    @cached_property
    def key(self):
        return self.coder.decode(self.encoded_key)

    @cached_property
    def value(self):
        return self.coder.decode(self.encoded_value, self.value_type)

    @cached_property
    def value_type(self):
        return type(self.value)

    @cached_property
    def encoded_key(self):
        return self.coder.encode(self.key)

    @cached_property
    def encoded_value(self):
        return self.coder.encode(self.value)

    @staticmethod
    def init(
        coder: KnowledgeCoder,
        key: Descriptive | None = None,
        value: Descriptive | None = None,
        value_type: type[Descriptive] | None = None,
        encoded_key: Embedding | None = None,
        encoded_value: Embedding | None = None,
    ):
        ak = AtomicKnowledge(coder)
        if key is not None:
            ak.key = key
        assert value is not None or value_type is not None
        if value is not None:
            ak.value = value
        if value_type is not None:
            ak.value_type = value_type
        if encoded_key is not None:
            ak.encoded_key = encoded_key
        if encoded_value is not None:
            ak.encoded_value = encoded_value
        return ak


@dataclass
class Knowledge:
    data: list[AtomicKnowledge]

    @property
    def format(self):
        return KnowledgeFormat(
            format=[
                AtomicFormat(
                    key=ak.key,
                    value_type=type(ak.value),
                    encoded_key=ak.encoded_key,
                    encoded_value_length=len(ak.encoded_value.data),
                )
                for ak in self.data
            ]
        )

    def distance_to(self, other: Knowledge) -> float:
        if len(self.data) != len(other.data):
            raise ValueError("Knowledge objects must have the same length for distance calculation.")
        return (
            sum(
                sum((a.encoded_value.data[i] - b.encoded_value.data[i]) ** 2 for i in range(len(a.encoded_value.data)))
                for a, b in zip(self.data, other.data)
            )
            ** 0.5
        )

    def to_numpy(self) -> np.ndarray:
        return np.concatenate([ak.encoded_value.to_numpy() for ak in self.data])

    def add(self, atomic_knowledge: AtomicKnowledge) -> None:
        self.data.append(atomic_knowledge)


@dataclass
class KnowledgeFactory:
    coder: KnowledgeCoder

    def from_dict(self, values: dict[str, Descriptive], embedding_size: int = None) -> Knowledge:
        coder = self.coder
        if embedding_size and self.coder.embedding_size != embedding_size:
            coder = self.coder.with_embedding_size(embedding_size)
        return Knowledge(
            data=[
                AtomicKnowledge.init(
                    coder,
                    key=key,
                    value=value,
                )
                for key, value in values.items()
            ],
        )

    def empty(self) -> Knowledge:
        return Knowledge(data=[])

    def from_numpy(self, values: np.ndarray, expected_format: KnowledgeFormat) -> Knowledge:
        knowledge = self.empty()
        offset = 0
        for af in expected_format.format:
            value = values[offset : offset + af.encoded_value_length]
            knowledge.add(
                AtomicKnowledge.init(
                    self.coder,
                    key=af.key,
                    value_type=af.value_type,
                    encoded_value=Embedding.from_numpy(value),
                )
            )
            offset += af.encoded_value_length
        return knowledge

    def from_format(self, expected_format: KnowledgeFormat) -> Knowledge:
        return Knowledge(
            [
                AtomicKnowledge.init(self.coder, key=af.key, encoded_value=np.zeros(af.encoded_value_length))
                for af in expected_format.format
            ]
        )


@dataclass
class KnowledgeService:
    response_limit: int

    def get(self, limit: int = None, name: str = None, expected_format: KnowledgeFormat = None) -> list[Knowledge]:
        limit = min(limit or self.response_limit, self.response_limit)
        return self._get_limited(limit, name, expected_format)

    def _get_limited(self, limit: int, name: str = None, expected_format: KnowledgeFormat = None) -> list[Knowledge]:
        raise NotImplementedError()
