import random
from dataclasses import dataclass

from knowledge import KnowledgeFactory


@dataclass
class Dataset:
    batch_size: int
    knowledge_factory: KnowledgeFactory

    def __iter__(self):
        for _ in range(self.batch_size):
            x = random.random()
            y = random.random()
            yield (
                self.knowledge_factory.from_list([x, y]),
                self.knowledge_factory.from_list([x + y, x - y, x * y, x / y]),
            )
