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
            return Embedding(
                tuple(ord(c) for c in value.ljust(self.embedding_size)[: self.embedding_size])
            )
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

    @property
    def n_items(self) -> int:
        return len(self.format)

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


@dataclass(frozen=True)
class Knowledge:
    data: list[AtomicKnowledge]

    @cached_property
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
        assert len(self.data) == len(other.data)
        return (
            sum(
                sum(
                    (a.encoded_value.data[i] - b.encoded_value.data[i]) ** 2
                    for i in range(len(a.encoded_value.data))
                )
                for a, b in zip(self.data, other.data)
            )
            ** 0.5
        )

    def to_numpy(self) -> np.ndarray:
        return np.array([ak.encoded_value.to_numpy() for ak in self.data])


@dataclass(frozen=True)
class PieceOfKnowledge:
    data: list[Knowledge]

    @staticmethod
    def empty():
        return PieceOfKnowledge([])

    @property
    def format(self):
        return self.data[0].format

    @property
    def size(self):
        return len(self.data)

    def merge(self, other: PieceOfKnowledge):
        if other.is_empty():
            return self
        if self.is_empty():
            return other
        return PieceOfKnowledge([Knowledge(x.data + y.data) for x, y in zip(self.data, other.data)])

    def is_empty(self):
        return not bool(self.data)

    def to_numpy(self):
        return np.array([x.to_numpy() for x in self.data])

    def distances_to(self, other: PieceOfKnowledge):
        return [x.distance_to(y) for x, y in zip(self.data, other.data)]

    def max_distance_to(self, other: PieceOfKnowledge):
        return max(self.distances_to(other))


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
        data = []
        offset = 0
        for af, value in zip(expected_format.format, values):
            data.append(
                AtomicKnowledge.init(
                    self.coder,
                    key=af.key,
                    value_type=af.value_type,
                    encoded_value=Embedding.from_numpy(value),
                )
            )
            offset += af.encoded_value_length
        return Knowledge(data)

    def from_format(self, expected_format: KnowledgeFormat) -> Knowledge:
        return Knowledge(
            [
                AtomicKnowledge.init(
                    self.coder,
                    key=af.key,
                    value_type=af.value_type,
                    encoded_value=Embedding(np.zeros(af.encoded_value_length)),
                )
                for af in expected_format.format
            ]
        )

    def from_numpy_batch(self, x: np.ndarray, expected_format: KnowledgeFormat):
        return PieceOfKnowledge([self.from_numpy(y, expected_format) for y in x])


@dataclass
class KnowledgeService:
    response_limit: int

    def get(
        self, limit: int = None, name: str = None, expected_format: KnowledgeFormat = None
    ) -> list[Knowledge]:
        limit = min(limit or self.response_limit, self.response_limit)
        return self._get_limited(limit, name, expected_format)

    def _get_limited(
        self, limit: int, name: str = None, expected_format: KnowledgeFormat = None
    ) -> list[Knowledge]:
        raise NotImplementedError()
