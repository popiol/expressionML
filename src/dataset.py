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
            operation = random.randrange(4)
            if operation == 0:
                result = x + y
            elif operation == 1:
                result = x - y
            elif operation == 2:
                result = x * y
            elif operation == 3:
                while abs(y) < 0.001:
                    y = random.random()
                result = x / y
            yield (
                self.knowledge_factory.from_dict({"x": x, "y": y, "operation": operation}),
                self.knowledge_factory.from_dict({"result": result}),
            )

    def get_batch(self):
        input_batch = []
        output_batch = []
        for inputs, outputs in self:
            input_batch.append(inputs)
            output_batch.append(outputs)
        return input_batch, output_batch
