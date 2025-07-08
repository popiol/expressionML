from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.agent import Agent
from src.knowledge import (
    KnowledgeFormat,
    PieceOfKnowledge,
)


@dataclass
class SupervisorAgent(Agent):
    eval_mode: bool = True

    def act(self, inputs: PieceOfKnowledge, expected_format: KnowledgeFormat) -> PieceOfKnowledge:
        print("act")
        assert not inputs.is_empty()
        self.eval_mode = True
        return self.run(inputs)

    def train(self, inputs: PieceOfKnowledge, outputs: PieceOfKnowledge):
        print("train agent")
        assert not inputs.is_empty()
        self.eval_mode = False
        calculated = self.run(inputs)
        assert calculated == outputs

    def approximate(
        self, func: Callable, inputs: PieceOfKnowledge, output_format: KnowledgeFormat, train: bool
    ) -> PieceOfKnowledge:
        model = self.get_model(inputs.format, output_format)
        if train:
            out_array = np.array([func(*[y.value for y in x.data]) for x in inputs.data])
            if out_array.ndim == 1:
                out_array = out_array.reshape(-1, 1)
            outputs = self.knowledge_factory.from_decoded_numpy(out_array, output_format)
            model.train(inputs, outputs)
            return outputs
        return model.predict(inputs)

    def run(self, inputs: PieceOfKnowledge) -> PieceOfKnowledge:
        raise NotImplementedError()
