"""Monte Carlo Tree Search with PUCT selection.

Works for single-player, two-player adversarial, multi-player general-sum,
and imperfect information (via determinization in Oracle).

Key optimizations:
  - Lazy expansion: transition only the selected child, not all children
  - Dirichlet noise at root: prevents local optima (AlphaZero)
  - Progressive widening: dynamically expand more actions as visits increase
  - Confidence estimation: quantifies how decisive the search result is
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Awaitable

from .core import Node, State, Action

if TYPE_CHECKING:
    from .oracle import Oracle


class BudgetExhausted(Exception):
    pass


@dataclass
class MCTSConfig:
    num_simulations: int = 40
    exploration: float = 1.41       # PUCT C constant
    max_depth: int = 8
    max_actions: int = 7            # cap on actions per node
    terminal_check_depth: int = 2
    # Dirichlet noise at root
    dirichlet_noise: bool = False
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    # Budget
    max_llm_calls: int = 0          # 0 = unlimited


class MCTS:
    """Monte Carlo Tree Search with PUCT selection and lazy expansion."""

    def __init__(self, oracle: Oracle, config: MCTSConfig | None = None):
        self.oracle = oracle
        self.cfg = config or MCTSConfig()

    async def search(
        self,
        root_state: State,
        root: Node | None = None,
        on_progress: Callable[[int, int, Node], Awaitable[None] | None] | None = None,
    ) -> Node:
        """Run MCTS. Pass an existing root for warm-start."""
        if root is None:
            root = Node(state=root_state)

        n = self.cfg.num_simulations
        for i in range(n):
            try:
                node, depth = await self._select_and_expand(root)

                if depth >= self.cfg.terminal_check_depth:
                    is_term, payoffs = await self.oracle.terminal(node.state)
                    if is_term and payoffs:
                        self._backpropagate(node, payoffs)
                        if on_progress:
                            await _maybe_await(on_progress(i + 1, n, root))
                        continue

                values = await self.oracle.evaluate(node.state)
                self._backpropagate(node, values)

                # Apply Dirichlet noise after first root expansion
                if i == 0 and self.cfg.dirichlet_noise and root.children:
                    self._apply_dirichlet_noise(root)

                if on_progress:
                    await _maybe_await(on_progress(i + 1, n, root))

            except BudgetExhausted:
                break

        return root

    # ── Result extraction ──────────────────────────────────

    def best_action(self, root: Node) -> Action:
        if not root.children:
            return Action(name="pass", description="No actions available")
        return max(root.children, key=lambda c: c.visits).action or Action(name="unknown")

    def principal_variation(self, root: Node) -> list[Action]:
        pv, node = [], root
        while node.children:
            best = max(node.children, key=lambda c: c.visits)
            if best.action:
                pv.append(best.action)
            node = best
        return pv

    def action_scores(self, root: Node) -> list[tuple[Action, float]]:
        total = max(sum(c.visits for c in root.children), 1)
        scores = [(c.action, c.visits / total) for c in root.children if c.action]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def confidence(self, root: Node) -> float:
        """Search confidence: 0=uniform visits, 1=one action dominates."""
        if len(root.children) < 2:
            return 1.0 if root.children else 0.0
        visits = [c.visits for c in root.children]
        total = sum(visits)
        if total == 0:
            return 0.0
        probs = [v / total for v in visits]
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        max_entropy = math.log(len(probs))
        return max(0.0, 1.0 - entropy / max_entropy) if max_entropy > 0 else 1.0

    # ── Internal: selection + lazy expansion ───────────────

    async def _select_and_expand(self, root: Node) -> tuple[Node, int]:
        node, depth = root, 0

        while depth < self.cfg.max_depth:
            if node._pending is None:
                actions = await self._oracle_generate(node.state)
                node._pending = actions[: self.cfg.max_actions]

            if node._pending:
                return await self._expand_one(node), depth + 1

            if node.children:
                node = self._puct_select(node)
                depth += 1
            else:
                break

        return node, depth

    async def _expand_one(self, node: Node) -> Node:
        pending = node._pending
        weights = [max(a.prior, 0.01) for a in pending]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        idx = random.choices(range(len(pending)), weights=probs, k=1)[0]
        action = pending.pop(idx)

        new_state = await self._oracle_simulate(node.state, action)
        child = Node(state=new_state, action=action, parent=node, prior=action.prior)
        node.children.append(child)
        return child

    def _puct_select(self, node: Node) -> Node:
        player = node.state.whose_turn
        sqrt_parent = math.sqrt(max(node.visits, 1))
        best_child, best_score = None, -math.inf

        for child in node.children:
            q = child.q(player)
            u = self.cfg.exploration * child.prior * sqrt_parent / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child or node

    def _backpropagate(self, node: Node | None, values: dict[str, float]) -> None:
        while node is not None:
            node.visits += 1
            for player, val in values.items():
                node.value_sum[player] = node.value_sum.get(player, 0.0) + val
            node = node.parent

    def _apply_dirichlet_noise(self, root: Node) -> None:
        n = len(root.children)
        if n == 0:
            return
        noise = _dirichlet([self.cfg.dirichlet_alpha] * n)
        eps = self.cfg.dirichlet_epsilon
        for child, eta in zip(root.children, noise):
            child.prior = (1 - eps) * child.prior + eps * eta

    # ── Oracle wrappers with budget tracking ──────────────

    async def _oracle_generate(self, state: State) -> list[Action]:
        self._check_budget()
        return await self.oracle.generate(state)

    async def _oracle_simulate(self, state: State, action: Action) -> State:
        self._check_budget()
        return await self.oracle.simulate(state, action)

    def _check_budget(self):
        if self.cfg.max_llm_calls > 0:
            if hasattr(self.oracle, "call_count") and self.oracle.call_count >= self.cfg.max_llm_calls:
                raise BudgetExhausted(f"Budget of {self.cfg.max_llm_calls} LLM calls exhausted")


def _dirichlet(alpha: list[float]) -> list[float]:
    """Simple Dirichlet sample using gamma distribution."""
    samples = [random.gammavariate(a, 1.0) for a in alpha]
    total = sum(samples) or 1.0
    return [s / total for s in samples]


async def _maybe_await(result):
    """Await if result is a coroutine."""
    if result is not None and hasattr(result, "__await__"):
        await result
