from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from src.agent import Agent
from src.knowledge import (
    KnowledgeFormat,
    PieceOfKnowledge,
)
from src.stats import Stats


@dataclass
class AdaBoostAgent(Agent):
    @cached_property
    def training_batches(self) -> list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        return []

    @cached_property
    def problem_classifier_format(self):
        return self.knowledge_factory.from_dict({"problem class": 0}, embedding_size=1).format

    def get_problem_classifier(self, input_format: KnowledgeFormat):
        return self.get_model(input_format, self.problem_classifier_format)

    def act(self, inputs: PieceOfKnowledge, expected_format: KnowledgeFormat) -> PieceOfKnowledge:
        print("act")
        assert not inputs.is_empty()
        bad = list(range(expected_format.size))
        out_array = np.zeros((inputs.size, expected_format.size))
        classes = np.zeros((inputs.size, 1))
        for _ in range(4):
            model = self.get_model(inputs.format, expected_format)
            out_array[bad] = model.predict(inputs.to_numpy()[bad])
            outputs = self.knowledge_factory.from_numpy_batch(out_array, expected_format)
            classifier = self.get_problem_classifier(inputs.format)
            classes[bad] = classifier.predict(inputs.to_numpy()[bad])
            bad = [yi for yi, y in enumerate(classes) if y < 0.5]
            print("# bad", len(bad))
            if not bad:
                break
            inputs = inputs.merge(outputs)
        return outputs

    def train(self, inputs: PieceOfKnowledge, outputs: PieceOfKnowledge):
        print("train agent")
        assert not inputs.is_empty()
        bad = list(range(inputs.size))
        out_array = np.zeros((inputs.size, outputs.format.size))
        classes = np.zeros((inputs.size, 1))
        for level in range(4):
            if len(self.training_batches) - 1 < level:
                self.training_batches.append([])
            training_batch = self.training_batches[level]
            training_batch.extend(
                list(zip(inputs.to_numpy()[bad], outputs.to_numpy()[bad], classes[bad]))
            )
            training_batch = training_batch[-inputs.size :]
            model = self.get_model(inputs.format, outputs.format)
            args = [np.array(x) for x in zip(*training_batch)]
            model.train(args[0], args[1])
            out_array[bad] = model.predict(inputs.to_numpy()[bad])
            predicted = self.knowledge_factory.from_numpy_batch(out_array, outputs.format)
            errors = np.array(outputs.distances_to(predicted))[bad]
            print("max error", max(errors))
            stats = Stats.from_batch(errors)
            classifier = self.get_problem_classifier(inputs.format)
            classes[bad] = [[int(error < stats.mean)] for error in errors]
            for i, (x, y, c) in enumerate(training_batch[-len(bad) :]):
                c[:] = classes[bad][i]
            args = [np.array(x) for x in zip(*training_batch)]
            classifier.train(args[0], args[2])
            if len(bad) <= 1:
                break
            bad = [yi for yi, y in enumerate(classes) if y < 0.5]
            if not bad:
                break
            inputs = inputs.merge(predicted)
            good = [yi for yi, y in enumerate(classes) if y > 0.5]
            print("sample", outputs.to_numpy()[good[0]])
