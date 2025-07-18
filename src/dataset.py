import random
from dataclasses import dataclass

import numpy as np

from src.coder import AdvancedCoder
from src.knowledge import Embedding, KnowledgeFactory, PieceOfKnowledge


@dataclass
class Dataset:
    batch_size: int
    knowledge_factory: KnowledgeFactory

    def arithmetic(self):
        for _ in range(self.batch_size):
            x = round(random.random() * 1000) / 1000
            y = round(random.random() * 1000) / 1000
            operation = 0  # random.randrange(2)
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
            n = random.randint(1, 10)
            bits = format(x, "b")
            bits = bits.zfill(64)[:64]
            shifted_bits = bits[n:] + bits[:n]
            result = int(shifted_bits, 2)
            yield (
                self.knowledge_factory.from_dict({"x": x, "shift": n}),
                self.knowledge_factory.from_dict({"result": result}),
            )

    def bitwise_operations(self):
        coder = AdvancedCoder(64)

        def to_bits(value):
            return [int(x) for x in coder.encode_float(value)]

        def bits_to_int(bits):
            return coder.decode_float(Embedding(bits))

        for _ in range(self.batch_size):
            while True:
                x = to_bits(random.randrange(1000))
                y = to_bits(random.randrange(1000))
                operation = random.randrange(3)
                if operation == 0:
                    result = bits_to_int(np.bitwise_and(x, y))  # AND
                elif operation == 1:
                    result = bits_to_int(np.bitwise_or(x, y))  # OR
                elif operation == 2:
                    result = bits_to_int(np.bitwise_xor(x, y))  # XOR
                    if any(np.bitwise_xor(x, y) != to_bits(result)):
                        continue
                x = bits_to_int(x)
                y = bits_to_int(y)
                yield (
                    self.knowledge_factory.from_dict({"x": x, "y": y, "operation": operation}),
                    self.knowledge_factory.from_dict({"result": result}),
                )
                break

    def one_hot(self):
        for _ in range(self.batch_size):
            x = random.randrange(64)
            y = [0] * 64
            y[x] = 1
            yield (
                self.knowledge_factory.from_dict({"x": x}),
                self.knowledge_factory.from_dict({f"result{yi}": y[yi] for yi in range(64)}),
            )

    def get_batch(self):
        input_batch = []
        output_batch = []
        for inputs, outputs in self.arithmetic():
            input_batch.append(inputs)
            output_batch.append(outputs)
        return PieceOfKnowledge(input_batch), PieceOfKnowledge(output_batch)
