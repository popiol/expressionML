from __future__ import annotations

from dataclasses import dataclass

from src.knowledge import (
    PieceOfKnowledge,
)
from src.supervisor_agent import SupervisorAgent


@dataclass
class BitwiseAddAgent(SupervisorAgent):
    def run(self, inputs: PieceOfKnowledge) -> PieceOfKnowledge:
        pass
