from __future__ import annotations

from dataclasses import dataclass

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
    key: Descriptive
    value: Descriptive
    encoded_key: Embedding
    encoded_value: Embedding


@dataclass
class Knowledge:
    data: list[AtomicKnowledge]
    capacity: int
    """The maximum number of atomic knowledge entries allowed in this knowledge base."""

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
    capacity: int

    def from_list(self, values: list[Descriptive]) -> Knowledge:
        return Knowledge(
            data=[
                AtomicKnowledge(
                    key=index,
                    value=value,
                    encoded_key=self.coder.encode(index),
                    encoded_value=self.coder.encode(value),
                )
                for index, value in enumerate(values)
            ],
            capacity=self.capacity,
        )

    def empty(self) -> Knowledge:
        return Knowledge(
            data=[],
            capacity=self.capacity,
        )

    def from_numpy(self, values: np.ndarray, expected_format: KnowledgeFormat) -> Knowledge:
        knowledge = self.empty()
        offset = 0
        for af in expected_format.format:
            value = values[offset : offset + af.encoded_value_length]
            knowledge.add(
                AtomicKnowledge(
                    key=af.key,
                    value=self.coder.decode(Embedding.from_numpy(value), af.value_type),
                    encoded_key=af.encoded_key,
                    encoded_value=Embedding.from_numpy(value),
                )
            )
            offset += af.encoded_value_length
        return knowledge
