import random
from dataclasses import dataclass

from src.knowledge import KnowledgeFactory


@dataclass
class Dataset:
    batch_size: int
    knowledge_factory: KnowledgeFactory

    def __iter__(self):
        for _ in range(self.batch_size):
            x = random.random()
            y = random.random()
            operation = random.randrange(2)
            if operation == 0:
                result = x + y
            elif operation == 1:
                result = x - y
            elif operation == 2:
                result = x * y
            elif operation == 3:
                result = x / y
            yield (
                self.knowledge_factory.from_dict({"x": x, "y": y, "operation": operation}),
                self.knowledge_factory.from_dict({"result": result}),
            )
