import random
from dataclasses import dataclass
from enum import Enum, auto

from src.agent import Agent
from src.dataset import Dataset
from src.knowledge import Knowledge, KnowledgeFactory
from src.lookup_coder import LookupCoder
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

    def evaluate(self, action: Knowledge, outputs: Knowledge):
        return -action.distance_to(outputs)

    def simulate(self):
        for inputs, outputs in self.dataset:
            action = self.agent.act(inputs, outputs.format)
            if random.random() < 0.01:
                # print("inputs:", inputs.data[0].value, inputs.data[1].value, inputs.data[2].value)
                print("outputs/pred:", outputs.data[0].value, action.data[0].value)
                # print("output", outputs.data[0].encoded_value)
                print("action", action.data[0].encoded_value)
            score = self.evaluate(action, outputs)
            yield inputs, outputs, action, score

    def train(self):
        print("train")
        self.agent.set_training_mode()
        for inputs, outputs, action, score in self.simulate():
            if self.train_mode == TrainMode.GROUND_TRUTH:
                self.agent.train(inputs, outputs)
            elif self.train_mode == TrainMode.FEEDBACK:
                self.agent.acknowledge_feedback(inputs, action, score)
            else:
                raise ValueError(f"Unknown train mode: {self.train_mode}")

    def test(self):
        print("test")
        self.agent.set_evaluation_mode()
        score_stats = Stats()
        for inputs, outputs, action, score in self.simulate():
            score_stats.add_single_value(score)
        print(f"Test score: {score_stats.mean}, {score_stats.min}")

    def run(self):
        for _ in range(self.max_iterations):
            self.train()
            self.test()


def main():
    embedding_size = 64
    knowledge_factory = KnowledgeFactory(coder=LookupCoder(embedding_size), capacity=1000)
    model_factory = MlModelFactory(batch_size=32)
    runner = Runner(
        dataset=Dataset(
            batch_size=32,
            knowledge_factory=knowledge_factory,
        ),
        agent=Agent.init(
            model_version="lstm",
            global_knowledge=None,
            use_memory=False,
            model_factory=model_factory,
            knowledge_factory=knowledge_factory,
            embedding_size=embedding_size,
            batch_size=32,
        ),
        max_iterations=100,
        train_mode=TrainMode.GROUND_TRUTH,
    )
    runner.run()


if __name__ == "__main__":
    with timer("Overall"):
        main()
