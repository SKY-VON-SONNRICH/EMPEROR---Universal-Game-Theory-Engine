"""Multi-AI collaboration — Oracle wrappers for ensemble evaluation.

The MCTS core never changes. Power comes from composing Oracle wrappers.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .core import Action, State

if TYPE_CHECKING:
    from .oracle import Oracle


class EnsembleOracle:
    """Aggregates multiple oracles: one generates, all evaluate."""

    def __init__(self, oracles: list[Oracle]):
        self.oracles = oracles
        self._primary = oracles[0]
        self.call_count = 0

    async def generate(self, state: State) -> list[Action]:
        self.call_count += 1
        return await self._primary.generate(state)

    async def evaluate(self, state: State) -> dict[str, float]:
        self.call_count += len(self.oracles)
        results = await asyncio.gather(*[o.evaluate(state) for o in self.oracles])
        all_players: set[str] = set()
        for r in results:
            all_players.update(r.keys())
        return {
            p: sum(r.get(p, 0.5) for r in results) / len(results)
            for p in all_players
        }

    async def simulate(self, state: State, action: Action) -> State:
        self.call_count += 1
        return await self._primary.simulate(state, action)

    async def terminal(self, state: State) -> tuple[bool, dict[str, float] | None]:
        self.call_count += 1
        return await self._primary.terminal(state)
