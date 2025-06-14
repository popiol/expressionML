from dataclasses import dataclass

from src.agent import Agent
from src.dataset import Dataset
from src.knowledge import Knowledge, KnowledgeEncoder, KnowledgeFactory


@dataclass
class Runner:
    max_iterations: int
    dataset: Dataset

    def evaluate(self, action: Knowledge, outputs: Knowledge):
        pass

    def simulate(self, agent: Agent):
        for inputs, outputs in self.dataset:
            action = agent.act(inputs)
            evaluation = self.evaluate(action, outputs)
            yield inputs, outputs, action, evaluation

    def train(self, agent: Agent):
        for inputs, outputs, action, evaluation in self.simulate(agent):
            pass

    def test(self, agent: Agent):
        for inputs, outputs, action, evaluation in self.simulate(agent):
            pass

    def run(self):
        agent = Agent()
        for _ in range(self.max_iterations):
            self.train(agent)
            self.test(agent)


if __name__ == "__main__":
    runner = Runner(
        max_iterations=100,
        dataset=Dataset(
            batch_size=32,
            knowledge_factory=KnowledgeFactory(encoder=KnowledgeEncoder()),
        ),
    )
    runner.run()
