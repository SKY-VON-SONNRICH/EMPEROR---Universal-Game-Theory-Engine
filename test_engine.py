"""Emperor test suite. Run: python test_engine.py"""

import asyncio
from emperor import MCTS, MCTSConfig, Action, State, Node, Strategy, text_tree, json_tree


class MockOracle:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.call_count = 0
        self._counts = {"gen": 0, "eval": 0, "sim": 0, "term": 0}

    async def generate(self, state):
        self._counts["gen"] += 1
        self.call_count += 1
        return [Action("left", prior=0.6), Action("right", prior=0.4)]

    async def evaluate(self, state):
        self._counts["eval"] += 1
        self.call_count += 1
        lefts = state.description.count("left")
        v = min(0.5 + 0.1 * lefts, 1.0)
        return {"A": v, "B": 1 - v}

    async def simulate(self, state, action):
        self._counts["sim"] += 1
        self.call_count += 1
        return state.advance(f"{state.description} {action.name}", action.name)

    async def terminal(self, state):
        self._counts["term"] += 1
        self.call_count += 1
        if len(state.history) >= self.max_depth:
            return True, {"A": 0.7, "B": 0.3}
        return False, None


def run(coro):
    return asyncio.run(coro)


def test_state_basics():
    s = State("start", players=["A", "B", "C"])
    assert s.whose_turn == "A"
    s2 = s.advance("next", "moved")
    assert s2.whose_turn == "B"
    assert len(s2.history) == 1
    s3 = s2.advance("next2", "moved2")
    assert s3.whose_turn == "C"
    # Test new fields
    s4 = State("test", perspective="limited view", metadata={"key": "val"})
    assert s4.perspective == "limited view"
    assert s4.metadata["key"] == "val"
    print("  ✓ test_state_basics")


def test_node_pending():
    n = Node(state=State("x"))
    assert n.is_leaf
    n._pending = [Action("a")]
    assert not n.is_leaf
    n._pending = []
    assert n.is_leaf
    print("  ✓ test_node_pending")


def test_mock_oracle_search():
    async def _run():
        oracle = MockOracle()
        mcts = MCTS(oracle, MCTSConfig(num_simulations=50, max_depth=4))
        state = State("root", players=["A", "B"])
        root = await mcts.search(state)
        best = mcts.best_action(root)
        assert best.name in ("left", "right")
        assert root.visits == 50
        assert len(root.children) > 0
        # Test confidence
        conf = mcts.confidence(root)
        assert 0 <= conf <= 1
        print(f"  ✓ test_mock_oracle_search (best={best.name}, conf={conf:.2f})")
    run(_run())


def test_lazy_expansion_efficiency():
    async def _run():
        oracle = MockOracle()
        mcts = MCTS(oracle, MCTSConfig(num_simulations=30, max_depth=4))
        state = State("root", players=["A", "B"])
        await mcts.search(state)
        # Lazy expansion: transitions should be <= simulations
        assert oracle._counts["sim"] <= 30, f"Too many transitions: {oracle._counts['sim']}"
        print(f"  ✓ test_lazy_expansion (sims={oracle._counts['sim']}, evals={oracle._counts['eval']})")
    run(_run())


def test_dirichlet_noise():
    async def _run():
        oracle = MockOracle()
        mcts = MCTS(oracle, MCTSConfig(num_simulations=20, max_depth=4, dirichlet_noise=True))
        state = State("root", players=["A", "B"])
        root = await mcts.search(state)
        # Should still work, children should have modified priors
        assert root.visits == 20
        print("  ✓ test_dirichlet_noise")
    run(_run())


def test_budget():
    async def _run():
        oracle = MockOracle()
        mcts = MCTS(oracle, MCTSConfig(num_simulations=100, max_depth=4, max_llm_calls=15))
        state = State("root", players=["A", "B"])
        root = await mcts.search(state)
        assert oracle.call_count <= 16  # small tolerance
        assert root.visits < 100  # should stop early
        print(f"  ✓ test_budget (calls={oracle.call_count}, visits={root.visits})")
    run(_run())


def test_tree_viz():
    async def _run():
        oracle = MockOracle()
        mcts = MCTS(oracle, MCTSConfig(num_simulations=20, max_depth=3))
        state = State("root", players=["A", "B"])
        root = await mcts.search(state)
        # text_tree
        txt = text_tree(root)
        assert "root" in txt
        # json_tree
        j = json_tree(root)
        assert "action" in j
        assert "description" in j
        assert "prior" in j
        assert j["visits"] > 0
        print("  ✓ test_tree_viz")
    run(_run())


def test_strategy_dataclass():
    s = Strategy(
        best_action=Action("x"),
        action_scores=[(Action("x"), 0.6)],
        principal_variation=[Action("x")],
        analysis="test",
        confidence=0.8,
        vulnerabilities=["risk1"],
    )
    assert s.confidence == 0.8
    assert len(s.vulnerabilities) == 1
    print("  ✓ test_strategy_dataclass")


def test_warm_start():
    async def _run():
        oracle = MockOracle()
        mcts = MCTS(oracle, MCTSConfig(num_simulations=10, max_depth=4))
        state = State("root", players=["A", "B"])
        root1 = await mcts.search(state)
        v1 = root1.visits
        # Warm-start: continue from existing tree
        root2 = await mcts.search(state, root=root1)
        assert root2.visits == v1 + 10
        print(f"  ✓ test_warm_start (v1={v1}, v2={root2.visits})")
    run(_run())


def test_progress_callback():
    async def _run():
        calls = []
        async def on_progress(sim, total, root):
            calls.append(sim)

        oracle = MockOracle()
        mcts = MCTS(oracle, MCTSConfig(num_simulations=10, max_depth=3))
        state = State("root", players=["A", "B"])
        await mcts.search(state, on_progress=on_progress)
        assert len(calls) == 10
        assert calls[-1] == 10
        print("  ✓ test_progress_callback")
    run(_run())


if __name__ == "__main__":
    print("Emperor Test Suite\n")
    test_state_basics()
    test_node_pending()
    test_mock_oracle_search()
    test_lazy_expansion_efficiency()
    test_dirichlet_noise()
    test_budget()
    test_tree_viz()
    test_strategy_dataclass()
    test_warm_start()
    test_progress_callback()
    print("\n✓ All tests passed!")
