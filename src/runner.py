from dataclasses import dataclass

import numpy as np

from src.agent import Agent
from src.coder import FloatCoder
from src.int_coder import IntCoder
from src.dataset import Dataset
from src.knowledge import KnowledgeFactory, PieceOfKnowledge
from src.ml_model import MlModelFactory
from src.stats import Stats
from src.utils import timer


@dataclass
class Runner:
    dataset: Dataset
    agent: Agent
    max_iterations: int
    supervised: bool

    def evaluate(self, actions: PieceOfKnowledge, outputs: PieceOfKnowledge):
        # return [-x for x in actions.distances_to(outputs)]
        return [
            -(sum((a.value - b.value) ** 2 for a, b in zip(x.data, y.data)) ** 0.5)
            for x, y in zip(actions.data, outputs.data)
        ]

    def simulate(self):
        inputs, outputs = self.dataset.get_batch()
        actions = self.agent.act(inputs, outputs.format)
        scores = self.evaluate(actions, outputs)
        worst = np.argmin(scores)
        print("inputs:", [x.value for x in inputs.data[worst].data])
        print("output:", [x.value for x in outputs.data[worst].data])
        print("action:", [x.value for x in actions.data[worst].data])
        for x in inputs.data[worst].data:
            print("input:", x.encoded_value.data)
        print("output", outputs.data[worst].data[0].encoded_value.data)
        print("action", [float(round(x)) for x in actions.data[worst].data[0].encoded_value.data])
        return inputs, outputs, actions, scores

    def train(self):
        print("train")
        if self.supervised:
            inputs, outputs, actions, scores = self.simulate()
            self.agent.train(inputs, outputs)
            # new_actions = self.agent.act(inputs, outputs.format)
            # new_scores = self.evaluate(new_actions, outputs)
            # new_inputs, new_outputs = self.agent.acknowledge_training_feedback(
            #     inputs, outputs, new_scores
            # )
            # new_actions = self.agent.act(new_inputs, outputs.format)
            # new_scores = self.evaluate(new_actions, new_outputs)
            # stats = Stats.from_batch(new_scores)
            # print(f"Train score: {stats.mean}, {stats.min}, {stats.count}")
        else:
            with self.agent.exploration_mode():
                inputs, outputs, actions, scores = self.simulate()
            self.agent.acknowledge_feedback(inputs, actions, scores)
        stats = Stats.from_batch(scores)
        print(f"Train score: {stats.mean}, {stats.min}")

    def test(self):
        print("test")
        inputs, outputs, actions, scores = self.simulate()
        stats = Stats.from_batch(scores)
        print(f"Test score: {stats.mean}, {stats.min}")

    def run(self):
        for _ in range(self.max_iterations):
            self.train()
            self.test()


def main():
    knowledge_factory = KnowledgeFactory(coder=FloatCoder(embedding_size=64))
    model_factory = MlModelFactory()
    runner = Runner(
        dataset=Dataset(
            batch_size=100,
            knowledge_factory=knowledge_factory,
        ),
        agent=Agent(
            model_version="v5",
            model_factory=model_factory,
            knowledge_factory=knowledge_factory,
        ),
        max_iterations=100,
        supervised=True,
    )
    runner.run()


if __name__ == "__main__":
    with timer("Overall"):
        main()
