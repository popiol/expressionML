from dataclasses import dataclass

from src.knowledge import Knowledge


@dataclass
class Agent:
    global_knowledge: Knowledge = None
    memory: Knowledge = None

    def act(self, communication: Knowledge) -> Knowledge:
        pass
