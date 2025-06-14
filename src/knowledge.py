from __future__ import annotations

from dataclasses import dataclass

Descriptive = str | int | float
Embedding = tuple[float, ...]


class KnowledgeEncoder:
    def encode(value: Descriptive) -> Embedding:
        if isinstance(value, str):
            return tuple(float(ord(c)) for c in value)
        elif isinstance(value, (int, float)):
            return (float(value),)
        else:
            raise ValueError(f"Unsupported type for encoding: {type(value)}")


@dataclass
class AtomicKnowledge:
    key: Descriptive
    value: Descriptive
    encoded_key: Embedding
    encoded_value: Embedding


class Knowledge:
    data: list[AtomicKnowledge]


class KnowledgeFactory:
    encoder: KnowledgeEncoder

    def from_list(self, values: list[Descriptive]) -> Knowledge:
        return Knowledge(
            [
                AtomicKnowledge(
                    key=index,
                    value=value,
                    encoded_key=self.encoder.encode(index),
                    encoded_value=self.encoder.encode(value),
                )
                for index, value in enumerate(values)
            ]
        )
