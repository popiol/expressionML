from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from src.knowledge import Knowledge, KnowledgeFactory, KnowledgeFormat, KnowledgeService
from src.ml_model import MlModel, MlModelFactory


@dataclass
class Agent:
    model_version: str
    global_knowledge: KnowledgeService | None
    memory: KnowledgeService | None
    use_short_memory: bool
    model_factory: MlModelFactory
    knowledge_factory: KnowledgeFactory
    embedding_size: int
    models: dict[tuple[KnowledgeFormat, KnowledgeFormat], MlModel]
    training_batch: list[tuple[float, Knowledge, Knowledge]]
    train_counter: int = 0

    @staticmethod
    def init(
        model_version: str,
        global_knowledge: KnowledgeService | None,
        use_memory: bool,
        use_short_memory: bool,
        model_factory: MlModelFactory,
        knowledge_factory: KnowledgeFactory,
        embedding_size: int,
    ) -> Agent:
        return Agent(
            model_version=model_version,
            model_factory=model_factory,
            models={},
            knowledge_factory=knowledge_factory,
            embedding_size=embedding_size,
            global_knowledge=global_knowledge,
            memory=KnowledgeService(0) if use_memory else None,
            use_short_memory=use_short_memory,
            training_batch=[],
        )

    def get_model(self, input_format: KnowledgeFormat, output_format: KnowledgeFormat) -> MlModel:
        key = (input_format, output_format)
        self.models[key] = self.models.get(
            key,
            self.model_factory.get_model(
                version=self.model_version,
                input_size=input_format.size,
                output_size=output_format.size,
            ),
        )
        return self.models[key]

    def set_training_mode(self):
        for model in self.models.values():
            model.set_training_mode()

    def set_evaluation_mode(self):
        for model in self.models.values():
            model.set_evaluation_mode()

    def act(self, inputs: list[Knowledge], expected_format: KnowledgeFormat) -> list[Knowledge]:
        print("act")
        assert inputs
        model = self.get_model(inputs[0].format, expected_format)
        all_inputs = [np.array([x.to_numpy() for x in inputs])]
        output_size = expected_format.size
        if self.use_short_memory:
            all_inputs.append(np.zeros((len(inputs), output_size)))
        prev_outputs = None
        for _ in range(10):
            outputs = model.predict(all_inputs)
            if not self.use_short_memory:
                break
            if prev_outputs is not None:
                diff = max(
                    self.knowledge_factory.from_numpy(x, expected_format).distance_to(
                        self.knowledge_factory.from_numpy(y, expected_format)
                    )
                    for x, y in zip(outputs, prev_outputs)
                )
                print("diff", diff)
                if diff < 0.001:
                    break
            prev_outputs = outputs
            all_inputs[1] = outputs
        return [self.knowledge_factory.from_numpy(x, expected_format) for x in outputs]

    def acknowledge_feedback(self, inputs: list[Knowledge], actions: list[Knowledge], scores: list[float]) -> None:
        print("acknowledge_feedback")
        self.train_counter += 1
        if len(self.training_batch) == 0:
            for x, y, s in zip(inputs, actions, scores):
                heapq.heappush(self.training_batch, (s, x, y))
        else:
            for x, y, s in zip(inputs, actions, scores):
                if s > self.training_batch[0][0]:
                    heapq.heappushpop(self.training_batch, (s, x, y))
        if self.train_counter >= 2:
            self.train_counter = 0
            print("worst score:", self.training_batch[0][0])
            model = self.get_model(inputs.format, actions.format)
            model.train(
                np.array([x[1].to_numpy() for x in self.training_batch]),
                np.array([x[2].to_numpy() for x in self.training_batch]),
            )

    def train(self, inputs: list[Knowledge], outputs: list[Knowledge]):
        print("train agent")
        assert inputs
        all_inputs = [np.array([x.to_numpy() for x in inputs])]
        output_size = outputs[0].format.size
        if self.use_short_memory:
            all_inputs.append(np.zeros((len(inputs), output_size)))
        all_outputs = np.array([x.to_numpy() for x in outputs])
        model = self.get_model(inputs[0].format, outputs[0].format)
        prev_predicted = None
        for _ in range(10):
            model.train(all_inputs, all_outputs)
            if not self.use_short_memory:
                break
            predicted = model.predict(all_inputs)
            error = max(
                self.knowledge_factory.from_numpy(x, outputs[0].format).distance_to(
                    self.knowledge_factory.from_numpy(y, outputs[0].format)
                )
                for x, y in zip(predicted, all_outputs)
            )
            print("error", error)
            if prev_predicted is not None:
                diff = max(
                    self.knowledge_factory.from_numpy(x, outputs[0].format).distance_to(
                        self.knowledge_factory.from_numpy(y, outputs[0].format)
                    )
                    for x, y in zip(predicted, prev_predicted)
                )
                print("diff", diff)
                if diff < 0.001:
                    break
            prev_predicted = predicted
            all_inputs[1] = predicted
