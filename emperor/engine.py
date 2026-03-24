"""Emperor — the main engine.

Usage:
    async with Emperor() as e:
        strategy = await e.analyze("situation", players=["You", "Rival"], goal="win")
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from .core import State, Node, Strategy
from .search import MCTS, MCTSConfig
from .oracle import LLMOracle, LLMConfig


@dataclass
class EmperorConfig:
    # LLM
    provider: str = "anthropic"
    model: str = ""
    api_key: str = ""
    temperature: float = 0.7
    reasoning: str = "standard"   # "standard" | "extended"
    # Search
    simulations: int = 40
    exploration: float = 1.41
    max_depth: int = 8
    max_actions: int = 7
    max_llm_calls: int = 0      # 0 = unlimited
    # Features
    reflect: bool = False
    dirichlet_noise: bool = True


class Emperor:
    """Universal game-theoretic decision engine.  MCTS × LLM Oracle."""

    def __init__(self, config: EmperorConfig | None = None):
        cfg = config or EmperorConfig()
        self._oracle = LLMOracle(LLMConfig(
            provider=cfg.provider,
            model=cfg.model,
            api_key=cfg.api_key,
            temperature=cfg.temperature,
            reasoning=cfg.reasoning,
        ))
        self._mcts = MCTS(self._oracle, MCTSConfig(
            num_simulations=cfg.simulations,
            exploration=cfg.exploration,
            max_depth=cfg.max_depth,
            max_actions=cfg.max_actions,
            max_llm_calls=cfg.max_llm_calls,
            dirichlet_noise=cfg.dirichlet_noise,
        ))
        self._cfg = cfg

    async def analyze(
        self,
        situation: str,
        *,
        players: list[str] | None = None,
        goal: str = "",
        perspective: str = "",
        simulations: int | None = None,
        root: Node | None = None,
        on_progress: Callable[[int, int, Node], Awaitable[None] | None] | None = None,
    ) -> Strategy:
        state = State(
            description=situation,
            players=players or ["You", "Opponent"],
            goal=goal,
            perspective=perspective,
        )

        if simulations is not None:
            self._mcts.cfg.num_simulations = simulations

        self._oracle.call_count = 0
        t0 = time.monotonic()

        # Run MCTS (warm-start if root provided)
        root = await self._mcts.search(state, root=root, on_progress=on_progress)
        self.last_root = root  # expose for tree serialization

        # Extract results
        best = self._mcts.best_action(root)
        pv = self._mcts.principal_variation(root)
        scores = self._mcts.action_scores(root)
        conf = self._mcts.confidence(root)

        # Optional: reflection pass
        vulns = []
        if self._cfg.reflect:
            vulns = await self._oracle.reflect(state, best, scores)

        # Synthesize analysis
        analysis = await self._oracle.synthesize(state, pv, scores)

        elapsed = time.monotonic() - t0

        return Strategy(
            best_action=best,
            action_scores=scores,
            principal_variation=pv,
            analysis=analysis,
            confidence=conf,
            vulnerabilities=vulns,
            stats={
                "simulations": root.visits,
                "tree_nodes": _count_nodes(root),
                "llm_calls": self._oracle.call_count,
                "elapsed_sec": round(elapsed, 2),
            },
        )

    async def chat(self, messages: list[dict]) -> str:
        """Direct conversation (no search)."""
        return await self._oracle.chat(messages)

    def run(self, situation: str, **kwargs) -> Strategy:
        return asyncio.run(self.analyze(situation, **kwargs))

    async def close(self):
        await self._oracle.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()


def _count_nodes(node: Node) -> int:
    return 1 + sum(_count_nodes(c) for c in node.children)
