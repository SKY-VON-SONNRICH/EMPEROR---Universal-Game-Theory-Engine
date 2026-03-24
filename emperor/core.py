"""Core data types — the entire domain model.

Every strategic scenario reduces to: States, Actions, and a Tree of possibilities.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    """A discrete choice a player can make."""
    name: str
    description: str = ""
    prior: float = 0.0  # LLM-estimated promise (0-1)


@dataclass
class State:
    """A snapshot of the strategic situation — in natural language."""
    description: str
    players: list[str] = field(default_factory=lambda: ["You", "Opponent"])
    current_player: int = 0
    history: list[str] = field(default_factory=list)
    goal: str = ""               # enables goal-conditioned (bidirectional) reasoning
    perspective: str = ""        # what the current player observes (imperfect info)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def whose_turn(self) -> str:
        return self.players[self.current_player % len(self.players)]

    def advance(self, description: str, action_desc: str, **meta) -> State:
        """Create successor state with updated history and next player."""
        return State(
            description=description,
            players=self.players,
            current_player=(self.current_player + 1) % len(self.players),
            history=[*self.history, f"{self.whose_turn}: {action_desc}"],
            goal=self.goal,
            metadata={**self.metadata, **meta},
        )


@dataclass
class Node:
    """A node in the Monte Carlo search tree."""
    state: State
    action: Action | None = None
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    visits: int = 0
    value_sum: dict[str, float] = field(default_factory=dict)
    prior: float = 1.0
    _pending: list[Action] | None = field(default=None, repr=False)

    def q(self, player: str) -> float:
        """Mean value for a player at this node."""
        return self.value_sum.get(player, 0.0) / max(self.visits, 1)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0 and not self._pending


@dataclass
class Strategy:
    """The engine's output — a complete strategic recommendation."""
    best_action: Action
    action_scores: list[tuple[Action, float]]  # sorted by score desc
    principal_variation: list[Action]           # best play sequence
    analysis: str                               # LLM-generated summary
    confidence: float = 0.0                     # 0=uncertain, 1=dominant
    vulnerabilities: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
