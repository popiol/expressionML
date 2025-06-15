from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from src.knowledge import Knowledge, KnowledgeFactory, KnowledgeFormat
from src.ml_model import MlModel, MlModelFactory
from src.stats import Stats


@dataclass
class Agent:
    model_version: str
    model_factory: MlModelFactory
    input_models: dict[KnowledgeFormat, MlModel]
    output_models: dict[KnowledgeFormat, MlModel]
    inner_model: MlModel
    combined_models: dict[tuple[KnowledgeFormat, KnowledgeFormat], MlModel]
    knowledge_factory: KnowledgeFactory
    embedding_size: int
    best_training_samples: list[tuple[float, Knowledge, Knowledge]]
    batch_size: int
    train_counter: int = 0
    score_stats: Stats = Stats()
    global_knowledge: Knowledge | None = None
    memory: Knowledge | None = None

    @staticmethod
    def init(
        model_version: str,
        global_knowledge: Knowledge | None,
        use_memory: bool,
        model_factory: MlModelFactory,
        knowledge_factory: KnowledgeFactory,
        embedding_size: int,
        batch_size: int,
    ) -> Agent:
        return Agent(
            model_version=model_version,
            model_factory=model_factory,
            input_models={},
            output_models={},
            inner_model=model_factory.get_model(
                version=model_version,
                input_size=3 * embedding_size,
                output_size=embedding_size,
            ),
            combined_models={},
            knowledge_factory=knowledge_factory,
            embedding_size=embedding_size,
            global_knowledge=global_knowledge,
            memory=knowledge_factory.empty() if use_memory else None,
            best_training_samples=[],
            batch_size=batch_size,
        )

    def get_input_model(self, input_format: KnowledgeFormat) -> MlModel:
        self.input_models[input_format] = self.input_models.get(
            input_format,
            self.model_factory.get_model(
                version=self.model_version,
                input_size=input_format.size,
                output_size=self.embedding_size,
            ),
        )
        return self.input_models[input_format]

    def get_output_model(self, output_format: KnowledgeFormat) -> MlModel:
        self.output_models[output_format] = self.output_models.get(
            output_format,
            self.model_factory.get_model(
                version=self.model_version,
                input_size=self.embedding_size,
                output_size=output_format.size,
            ),
        )
        return self.output_models[output_format]

    def get_input_models(self, input_format: KnowledgeFormat) -> list[MlModel]:
        models = []
        models.append(self.get_input_model(input_format))
        if self.global_knowledge is not None:
            models.append(self.get_input_model(self.global_knowledge.format))
        if self.memory is not None:
            models.append(self.get_input_model(self.memory.format))
        return models

    def get_combined_model(self, input_format: KnowledgeFormat, output_format: KnowledgeFormat) -> MlModel:
        key = (input_format, output_format)
        self.combined_models[key] = self.combined_models.get(
            key,
            self.model_factory.combine_models(
                input_models=self.get_input_models(input_format),
                inner_model=self.inner_model,
                output_model=self.get_output_model(output_format),
            ),
        )
        return self.combined_models[key]

    def set_training_mode(self):
        for model in self.combined_models.values():
            model.set_training_mode()

    def set_evaluation_mode(self):
        for model in self.combined_models.values():
            model.set_evaluation_mode()

    def act(self, inputs: Knowledge, expected_format: KnowledgeFormat) -> Knowledge:
        combined_model = self.get_combined_model(inputs.format, expected_format)
        outputs = combined_model.predict(inputs.to_numpy())
        return self.knowledge_factory.from_numpy(outputs, expected_format)

    def acknowledge_feedback(self, inputs: Knowledge, action: Knowledge, score: float) -> None:
        self.train_counter += 1
        if len(self.best_training_samples) < self.batch_size:
            heapq.heappush(self.best_training_samples, (score, inputs, action))
        elif score > self.best_training_samples[0][0]:
            heapq.heappop(self.best_training_samples)
            heapq.heappush(self.best_training_samples, (score, inputs, action))
        if len(self.best_training_samples) >= self.batch_size and self.train_counter == 2 * self.batch_size:
            self.train_counter = 0
            print("worst score:", self.best_training_samples[0][0])
            combined_model = self.get_combined_model(inputs.format, action.format)
            combined_model.train_batch(
                np.array([x[1].to_numpy() for x in self.best_training_samples]),
                np.array([x[2].to_numpy() for x in self.best_training_samples]),
            )

    def train(self, inputs: Knowledge, outputs: Knowledge):
        combined_model = self.get_combined_model(inputs.format, outputs.format)
        combined_model.train(inputs.to_numpy(), outputs.to_numpy())
