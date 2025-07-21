from __future__ import annotations

import heapq
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property

from src.knowledge import (
    Knowledge,
    KnowledgeFactory,
    KnowledgeFormat,
    KnowledgeService,
    PieceOfKnowledge,
)
from src.ml_model import MlModel, MlModelFactory


@dataclass
class Agent:
    model_version: str
    model_factory: MlModelFactory
    knowledge_factory: KnowledgeFactory
    global_knowledge: KnowledgeService | None = None
    memory: KnowledgeService | None = None
    _exploration_mode: bool = False
    train_counter: int = 0

    @cached_property
    def models(self) -> dict[tuple[KnowledgeFormat, KnowledgeFormat], MlModel]:
        return {}

    @cached_property
    def training_batch(self) -> list[tuple[float, Knowledge, Knowledge]]:
        return []

    @contextmanager
    def exploration_mode(self):
        self._exploration_mode = True
        yield self
        self._exploration_mode = False

    def get_model(self, input_format: KnowledgeFormat, output_format: KnowledgeFormat) -> MlModel:
        key = (input_format, output_format)
        if key not in self.models:
            self.models[key] = self.model_factory.get_model(
                version=self.model_version,
                in_objects=input_format.n_items,
                out_objects=output_format.n_items,
                n_features=self.knowledge_factory.coder.embedding_size,
            )
            print(self.models[key].model.summary())
        if self._exploration_mode:
            return self.model_factory.get_model_with_noise(self.models[key])
        return self.models[key]

    def act(self, inputs: PieceOfKnowledge, expected_format: KnowledgeFormat) -> PieceOfKnowledge:
        print("act")
        assert not inputs.is_empty()
        model = self.get_model(inputs.format, expected_format)
        return self.knowledge_factory.from_numpy_batch(
            model.predict(inputs.to_numpy()), expected_format
        )

    def train(self, inputs: PieceOfKnowledge, outputs: PieceOfKnowledge):
        print("train agent")
        assert not inputs.is_empty()
        model = self.get_model(inputs.format, outputs.format)
        model.train(inputs.to_numpy(), outputs.to_numpy())

    def acknowledge_feedback(
        self, inputs: PieceOfKnowledge, actions: PieceOfKnowledge, scores: list[float]
    ):
        print("acknowledge_feedback")
        self.train_counter += 1
        if len(self.training_batch) == 0:
            for x, y, s in zip(inputs, actions, scores):
                heapq.heappush(self.training_batch, (s, x, y))
        else:
            for x, y, s in zip(inputs.data, actions.data, scores):
                if s > self.training_batch[0][0]:
                    heapq.heappushpop(self.training_batch, (s, x, y))
        if self.train_counter >= 2:
            self.train_counter = 0
            print("worst score:", self.training_batch[0][0])
            self.train(
                PieceOfKnowledge([x for _, x, _ in self.training_batch]),
                PieceOfKnowledge([y for _, _, y in self.training_batch]),
            )
