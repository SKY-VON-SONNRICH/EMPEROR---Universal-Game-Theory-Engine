"""Game Oracle — for formal games with explicit rules.

Informal scenarios → LLMOracle (natural language)
Formal games → GameOracle subclass (coded rules)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from .core import Action, State


class GameOracle(ABC):
    """Abstract base for deterministic game rules. Implement 4 methods."""

    @abstractmethod
    def legal_actions(self, state: State) -> list[Action]: ...

    @abstractmethod
    def apply_action(self, state: State, action: Action) -> State: ...

    @abstractmethod
    def score(self, state: State) -> dict[str, float]: ...

    @abstractmethod
    def is_over(self, state: State) -> bool: ...

    # ── Oracle protocol implementation ─────────────────────

    async def generate(self, state: State) -> list[Action]:
        actions = self.legal_actions(state)
        n = len(actions) or 1
        for a in actions:
            if a.prior == 0.0:
                a.prior = 1.0 / n
        return actions

    async def evaluate(self, state: State) -> dict[str, float]:
        return self.score(state)

    async def simulate(self, state: State, action: Action) -> State:
        return self.apply_action(state, action)

    async def terminal(self, state: State) -> tuple[bool, dict[str, float] | None]:
        if self.is_over(state):
            return True, self.score(state)
        return False, None
