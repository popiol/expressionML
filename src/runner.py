from dataclasses import dataclass
from enum import Enum, auto

from src.agent import Agent
from src.dataset import Dataset
from src.knowledge import Knowledge, KnowledgeCoder, KnowledgeFactory
from src.ml_model import MlModelFactory


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
        return action.distance_to(outputs)

    def simulate(self):
        for inputs, outputs in self.dataset:
            action = self.agent.act(inputs, outputs.format)
            score = self.evaluate(action, outputs)
            yield inputs, outputs, action, score

    def train(self):
        self.agent.set_training_mode()
        for inputs, outputs, action, score in self.simulate():
            if self.train_mode == TrainMode.GROUND_TRUTH:
                self.agent.train(inputs, outputs)
            elif self.train_mode == TrainMode.FEEDBACK:
                self.agent.acknowledge_feedback(inputs, action, score)
            else:
                raise ValueError(f"Unknown train mode: {self.train_mode}")

    def test(self):
        self.agent.set_evaluation_mode()
        for inputs, outputs, action, score in self.simulate():
            print("Score:", score)

    def run(self):
        for _ in range(self.max_iterations):
            self.train()
            self.test()


if __name__ == "__main__":
    knowledge_factory = KnowledgeFactory(coder=KnowledgeCoder())
    model_factory = MlModelFactory()
    runner = Runner(
        dataset=Dataset(
            batch_size=32,
            knowledge_factory=knowledge_factory,
        ),
        agent=Agent.init(
            model_version="v1",
            global_knowledge=None,
            use_memory=False,
            model_factory=model_factory,
            knowledge_factory=knowledge_factory,
        ),
        max_iterations=100,
        train_mode=TrainMode.FEEDBACK,
    )
    runner.run()
