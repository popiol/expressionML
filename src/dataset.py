import random
from dataclasses import dataclass

from src.knowledge import KnowledgeFactory


@dataclass
class Dataset:
    batch_size: int
    knowledge_factory: KnowledgeFactory

    def arithmetic(self):
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
                while abs(y) < 0.001:
                    y = random.random()
                result = x / y
            yield (
                self.knowledge_factory.from_dict({"x": x, "y": y, "operation": operation}),
                self.knowledge_factory.from_dict({"result": result}),
            )

    def identity(self):
        for _ in range(self.batch_size):
            x = random.random()
            yield (
                self.knowledge_factory.from_dict({"x": x}),
                self.knowledge_factory.from_dict({"result": x}),
            )

    def cyclic_shift(self):
        for _ in range(self.batch_size):
            x = random.randrange(1000)
            n = 1  # random.randint(1, 10)
            bits = format(x, "b")
            bits = bits.zfill(64)[:64]
            shifted_bits = bits[n:] + bits[:n]
            result = int(shifted_bits, 2)
            yield (
                self.knowledge_factory.from_dict({"x": x, "shift": n}),
                self.knowledge_factory.from_dict({"result": result}),
            )

    def get_batch(self):
        input_batch = []
        output_batch = []
        for inputs, outputs in self.cyclic_shift():
            input_batch.append(inputs)
            output_batch.append(outputs)
        return input_batch, output_batch
