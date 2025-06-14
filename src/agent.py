from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.knowledge import EMBEDDING_SIZE, Knowledge, KnowledgeFactory, KnowledgeFormat
from src.ml_model import MlModel, MlModelFactory
from src.stats import Stats


@dataclass
class Agent:
    model_version: str
    model_factory: MlModelFactory
    input_models: dict[KnowledgeFormat, MlModel]
    output_models: dict[KnowledgeFormat, MlModel]
    inner_model: MlModel
    knowledge_factory: KnowledgeFactory
    stats: Stats = Stats()
    global_knowledge: Knowledge | None = None
    memory: Knowledge | None = None

    @staticmethod
    def init(
        model_version: str,
        global_knowledge: Knowledge | None,
        use_memory: bool,
        model_factory: MlModelFactory,
        knowledge_factory: KnowledgeFactory,
    ) -> Agent:
        return Agent(
            model_version=model_version,
            model_factory=model_factory,
            input_models={},
            output_models={},
            inner_model=model_factory.get_model(
                version=model_version,
                input_size=3 * EMBEDDING_SIZE,
                output_size=EMBEDDING_SIZE,
            ),
            knowledge_factory=knowledge_factory,
            global_knowledge=global_knowledge,
            memory=Knowledge.empty() if use_memory else None,
        )

    def get_input_model(self, input_format: KnowledgeFormat) -> MlModel:
        self.input_models[input_format] = self.input_models.get(
            input_format,
            self.model_factory.get_model(
                version=self.model_version,
                input_size=input_format.size,
                output_size=EMBEDDING_SIZE,
            ),
        )
        return self.input_models[input_format]

    def get_output_model(self, output_format: KnowledgeFormat) -> MlModel:
        self.output_models[output_format] = self.output_models.get(
            output_format,
            self.model_factory.get_model(
                version=self.model_version,
                input_size=EMBEDDING_SIZE,
                output_size=output_format.size,
            ),
        )
        return self.output_models[output_format]

    def set_training_mode(self):
        for model in self.input_models.values():
            model.set_training_mode()
        for model in self.output_models.values():
            model.set_training_mode()
        self.inner_model.set_training_mode()

    def set_evaluation_mode(self):
        for model in self.input_models.values():
            model.set_evaluation_mode()
        for model in self.output_models.values():
            model.set_evaluation_mode()
        self.inner_model.set_evaluation_mode()

    def get_inner_inputs(self, inputs: Knowledge) -> list[np.ndarray]:
        inner_inputs = []
        input_model = self.get_input_model(inputs.format)
        x1 = input_model.predict(inputs.to_numpy())
        inner_inputs.append(x1)
        if self.global_knowledge is not None:
            global_knowledge_model = self.get_input_model(self.global_knowledge.format)
            x2 = global_knowledge_model.predict(self.global_knowledge.to_numpy())
            inner_inputs.append(x2)
        if self.memory is not None:
            memory_model = self.get_input_model(self.memory.format)
            x3 = memory_model.predict(self.memory.to_numpy())
            inner_inputs.append(x3)
        return inner_inputs

    def get_outputs(self, inner_inputs: list[np.ndarray], expected_format: KnowledgeFormat) -> np.ndarray:
        output_model = self.get_output_model(expected_format)
        x = np.concatenate(inner_inputs, axis=1)
        return output_model.predict(x)

    def act(self, inputs: Knowledge, expected_format: KnowledgeFormat) -> Knowledge:
        inner_inputs = self.get_inner_inputs(inputs)
        outputs = self.get_outputs(inner_inputs, expected_format)
        return self.knowledge_factory.from_numpy(outputs, expected_format)

    def train_input_models(self, inputs: Knowledge, inner_inputs: list[np.ndarray]) -> None:
        input_model = self.get_input_model(inputs.format)
        input_model.train(inputs.to_numpy(), inner_inputs[0])
        if self.global_knowledge is not None:
            global_knowledge_model = self.get_input_model(self.global_knowledge.format)
            global_knowledge_model.train(self.global_knowledge.to_numpy(), inner_inputs[1])
        if self.memory is not None:
            memory_model = self.get_input_model(self.memory.format)
            memory_model.train(self.memory.to_numpy(), inner_inputs[2])

    def train_output_model(self, inner_inputs: list[np.ndarray], action: Knowledge) -> None:
        x = np.concatenate(inner_inputs, axis=1)
        output_model = self.get_output_model(action.format)
        output_model.train(x, action.to_numpy())

    def acknowledge_feedback(self, inputs: Knowledge, action: Knowledge, score: float) -> None:
        self.stats.add_single_value(score)
        if score > self.stats.mean + self.stats.std:
            inner_inputs = self.get_inner_inputs(inputs)
            self.train_input_models(inputs, inner_inputs)
            self.train_output_model(inner_inputs, action)

    def train(self, inputs: Knowledge, outputs: Knowledge):
        inner_inputs = self.get_inner_inputs(inputs)
        self.train_input_models(inputs, inner_inputs)
        self.train_output_model(inner_inputs, outputs)
