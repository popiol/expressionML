from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.knowledge import PieceOfKnowledge
from src.supervisor_agent import SupervisorAgent


@dataclass
class BitwiseAddAgent(SupervisorAgent):
    def run(self, inputs: PieceOfKnowledge, expected_format: KnowledgeFormat) -> PieceOfKnowledge:
        def get_significand(x):
            return x[:-4]

        def get_exponent(x):
            return [x[-4] + x[-3] * 1 / 2 + x[-2] * 1 / 4 + x[-1]]

        def get_common_exponent(x, y):
            return [min(x[0], y[0])]

        def get_diff(x, y):
            return [a - b for a, b in zip(x, y)]

        def shift(x, n):
            n = round(n[0])
            size = len(x)
            return [0] * (size - n) + x[max(0, n - size) :] + [0] * min(n, 2 * size)

        def add(x, y):
            bit4 = 0
            result = []
            for bit1, bit2 in zip(reversed(x), reversed(y)):
                bit3 = bit1 + bit2 - 2 * bit1 * bit2  # 1 xor 2
                bit3 = bit3 + bit4 - 2 * bit3 * bit4  # 3 xor 4
                bit4 = (
                    bit1 * bit2 + bit1 * bit4 + bit2 * bit4 - bit1 * bit2 * bit4
                )  # (1 and 2) or (1 and 4) or (2 and 4)
                result.append(bit3)
            return list(reversed(result))

        outputs = []

        for record in inputs.data:
            x = record.data[0].encoded_value.data
            y = record.data[1].encoded_value.data
            # x_s = get_significand(x)
            # x_e = get_exponent(x)
            # y_s = get_significand(y)
            # y_e = get_exponent(y)
            # exponent = get_common_exponent(x_e, y_e)
            # x_shift_n = get_diff(x_e, exponent)
            # x_s = shift(x_s, x_shift_n)
            # y_shift_n = get_diff(y_e, exponent)
            # y_s = shift(y_s, y_shift_n)
            result = add(x, y)
            outputs.append(result)

        return self.knowledge_factory.from_numpy_batch(np.array(outputs), expected_format)
