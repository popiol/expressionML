from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from src.agent import Agent
from src.coder import AdvancedCoder
from src.dataset import Dataset
from src.knowledge import Knowledge, KnowledgeFactory
from src.ml_model import MlModelFactory
from src.stats import Stats
from src.utils import timer


class TrainMode(Enum):
    GROUND_TRUTH = auto()
    FEEDBACK = auto()


@dataclass
class Runner:
    dataset: Dataset
    agent: Agent
    max_iterations: int
    train_mode: TrainMode

    def evaluate(self, actions: list[Knowledge], outputs: list[Knowledge]):
        # return -action.distance_to(outputs)
        return [-abs(x.data[0].value - y.data[0].value) for x, y in zip(actions, outputs)]

    def simulate(self):
        inputs, outputs = self.dataset.get_batch()
        actions = self.agent.act(inputs, outputs[0].format)
        scores = self.evaluate(actions, outputs)
        worst = np.argmin(scores)
        print("inputs:", [x.value for x in inputs[worst].data])
        print("outputs/pred:", outputs[worst].data[0].value, actions[worst].data[0].value)
        print("output", outputs[worst].data[0].encoded_value.data)
        print("action", [float(round(x)) for x in actions[worst].data[0].encoded_value.data])
        return inputs, outputs, actions, scores

    def train(self):
        print("train")
        stats = Stats()
        if self.train_mode == TrainMode.GROUND_TRUTH:
            inputs, outputs, actions, scores = self.simulate()
            self.agent.train(inputs, outputs)
        elif self.train_mode == TrainMode.FEEDBACK:
            self.agent.set_training_mode()
            inputs, outputs, actions, scores = self.simulate()
            self.agent.acknowledge_feedback(inputs, actions, scores)
        else:
            raise ValueError(f"Unknown train mode: {self.train_mode}")
        stats.add_batch(scores)
        print(f"Train score: {stats.mean}, {stats.min}")

    def test(self):
        print("test")
        self.agent.set_evaluation_mode()
        stats = Stats()
        inputs, outputs, actions, scores = self.simulate()
        stats.add_batch(scores)
        print(f"Test score: {stats.mean}, {stats.min}")

    def run(self):
        for _ in range(self.max_iterations):
            self.train()
            self.test()


def main():
    embedding_size = 64
    knowledge_factory = KnowledgeFactory(coder=AdvancedCoder(embedding_size))
    model_factory = MlModelFactory()
    runner = Runner(
        dataset=Dataset(
            batch_size=100,
            knowledge_factory=knowledge_factory,
        ),
        agent=Agent.init(
            model_version="v4",
            global_knowledge=None,
            use_memory=False,
            use_short_memory=False,
            model_factory=model_factory,
            knowledge_factory=knowledge_factory,
            embedding_size=embedding_size,
        ),
        max_iterations=100,
        train_mode=TrainMode.GROUND_TRUTH,
    )
    runner.run()


if __name__ == "__main__":
    with timer("Overall"):
        main()
