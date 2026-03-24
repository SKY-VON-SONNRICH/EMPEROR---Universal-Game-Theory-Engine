# Emperor — Architecture Design Document

**Universal Strategic Reasoning Engine: MCTS × LLM = Game Theory for Everything**

---

## 1. Design Philosophy

### Core Principle

> Every strategic situation in the universe reduces to: **States, Actions, and a Search over possibilities.**

An LLM is a universal world model — it encodes knowledge about business, games, psychology, economics, warfare, and every domain humans have written about. MCTS is a mathematically rigorous search algorithm with convergence guarantees. Their product is a universal strategy engine.

### Design Axioms

1. **Oracle is the only abstraction boundary.** MCTS knows nothing about domains. The Oracle knows nothing about search. This single interface is the entire architecture.
2. **LLM replaces three neural networks.** In AlphaZero: policy network → `oracle.generate()`, value network → `oracle.evaluate()`, dynamics model → `oracle.simulate()`. One LLM call replaces what took millions of self-play games to train.
3. **Composition over complexity.** Advanced features (multi-AI, reflection, adversarial testing) are Oracle wrappers. The MCTS core never changes. Power comes from composing simple pieces.
4. **Lazy everything.** Every LLM call costs time and money. Never call the Oracle unless the search algorithm demands it. Lazy expansion cuts API calls by ~75%.
5. **Defaults are production-ready.** Turn on Emperor with zero configuration and get good results. Every optional feature has a sensible default of OFF.

---

## 2. Architecture Overview

```
                    ┌──────────────────────┐
                    │    Natural Language   │
                    │   (user describes     │
                    │    situation + goal)   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │      Emperor         │
                    │    (orchestrator)     │
                    │                      │
                    │  1. Frame            │
                    │  2. Search (MCTS)    │
                    │  3. Reflect*         │
                    │  4. Synthesize       │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │  LLM Oracle    │ │  Game Oracle│ │  Council*   │
    │ (GPT/Claude)   │ │  (coded     │ │ (multi-AI   │
    │                │ │   rules)    │ │  ensemble)  │
    │ • generate     │ │ • legal_act │ │ wraps any   │
    │ • evaluate     │ │ • apply_act │ │ Oracle(s)   │
    │ • simulate     │ │ • score     │ │             │
    │ • terminal     │ │ • terminal  │ │             │
    └────────────────┘ └─────────────┘ └─────────────┘

    * = optional, OFF by default
```

### File Structure

```
emperor/
├── __init__.py       # Public API exports
├── core.py           # State, Action, Node, Strategy  (~80 lines)
├── oracle.py         # Oracle protocol + LLMOracle    (~320 lines)
├── search.py         # MCTS with PUCT + improvements  (~250 lines)
├── engine.py         # Emperor orchestrator            (~180 lines)
├── game.py           # GameOracle for formal games     (~75 lines)
├── council.py        # Multi-AI collaboration          (~100 lines)
├── viz.py            # Tree visualization              (~110 lines)
├── cli.py            # CLI interface                   (~80 lines)
├── tictactoe.py      # Example: formal game
└── test_engine.py    # Test suite
```

**Total: ~1,200 lines of core code. 1 dependency (`httpx`).**

---

## 3. Core Data Types (`core.py`)

### Action

```python
@dataclass
class Action:
    """A discrete choice a player can make."""
    name: str                    # Short identifier: "raise_price", "cooperate"
    description: str = ""        # Natural language explanation
    prior: float = 0.0           # LLM-estimated promise [0, 1], sums to ~1.0
```

**Design decision**: Priors are essential. They guide MCTS toward promising branches (PUCT formula), reducing the search space exponentially. Without priors, MCTS would waste simulations on clearly bad actions.

### State

```python
@dataclass
class State:
    """A snapshot of the strategic situation — in natural language."""
    description: str                   # The situation as natural language
    players: list[str]                 # All players (default: ["You", "Opponent"])
    current_player: int = 0            # Index into players
    history: list[str] = []            # Past moves as strings
    goal: str = ""                     # Enables goal-conditioned (bidirectional) reasoning
    perspective: str = ""              # What the current player observes (for imperfect info)
    metadata: dict[str, Any] = {}      # Extensible: milestones, constraints, custom data

    @property
    def whose_turn(self) -> str:
        return self.players[self.current_player % len(self.players)]

    def advance(self, description: str, action_desc: str, **meta) -> State:
        """Create successor state with next player and updated history."""
        return State(
            description=description,
            players=self.players,
            current_player=(self.current_player + 1) % len(self.players),
            history=[*self.history, f"{self.whose_turn}: {action_desc}"],
            goal=self.goal,
            metadata={**self.metadata, **meta},
        )
```

**New field: `perspective`** — For imperfect information scenarios. When set, the Oracle evaluation considers that the current player only sees this view of the world, not the full `description`. This is how poker hands, hidden agendas, and private information are modeled.

**New field: `metadata`** — Extensible dict for domain-specific data. Goal milestones, constraints, round counters, etc. Keeps the core State slim while allowing arbitrary extensions.

**Design decision**: Natural language states rather than formal representations. This is the key innovation — any situation describable in language can be analyzed. The cost is approximate (LLM-based) rather than exact evaluation, but the universality gain is infinite.

### Node

```python
@dataclass
class Node:
    """A node in the Monte Carlo search tree."""
    state: State
    action: Action | None = None        # Action that led here (None for root)
    parent: Node | None = None
    children: list[Node] = []
    visits: int = 0
    value_sum: dict[str, float] = {}    # {player_name: cumulative_score}
    prior: float = 1.0
    _pending: list[Action] | None = None   # Lazy expansion queue
    _virtual_loss: int = 0                 # For concurrent search

    def q(self, player: str) -> float:
        """Mean value for a player, accounting for virtual losses."""
        effective_visits = max(self.visits + self._virtual_loss, 1)
        effective_sum = self.value_sum.get(player, 0.0)
        return effective_sum / effective_visits

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0 and not self._pending
```

**New field: `_virtual_loss`** — When a node is being evaluated concurrently, its Q-value is temporarily reduced. This prevents multiple concurrent simulations from all going down the same path. Standard technique from AlphaGo, adapted for async LLM calls.

### Strategy

```python
@dataclass
class Strategy:
    """The engine's output — a complete strategic recommendation."""
    best_action: Action
    action_scores: list[tuple[Action, float]]   # Sorted by score desc
    principal_variation: list[Action]             # Best play sequence
    analysis: str                                 # LLM-generated strategic analysis
    confidence: float = 0.0                       # Search confidence [0, 1]
    vulnerabilities: list[str] = []               # Identified weaknesses (from reflection)
    stats: dict = {}                              # Search statistics
```

**New field: `confidence`** — Derived from search statistics. If one action dominates with 80%+ of visits, confidence is high. If visits are evenly split, confidence is low. Formula: `confidence = max_visit_proportion - 1/num_actions` normalized to [0, 1].

**New field: `vulnerabilities`** — Populated by the optional reflection pass. Lists specific weaknesses or counter-strategies the opponent might exploit.

---

## 4. Oracle Protocol (`oracle.py`)

### The Protocol

```python
@runtime_checkable
class Oracle(Protocol):
    """The universal world model interface.

    Any intelligence source — LLM, coded rules, neural network, human —
    can power the search engine by implementing these 4 methods.
    """
    async def generate(self, state: State) -> list[Action]:
        """Generate candidate actions with prior probabilities.

        Returns 3-7 actions sorted by promise. Priors should sum to ~1.0.
        This is the POLICY function — it says what's worth trying.
        """
        ...

    async def evaluate(self, state: State) -> dict[str, float]:
        """Score the position for each player.

        Returns {player_name: score} where 0.0 = worst, 0.5 = neutral, 1.0 = best.
        This is the VALUE function — it says how good things look.
        """
        ...

    async def simulate(self, state: State, action: Action) -> State:
        """Predict the outcome of taking an action.

        Returns the successor state. This is the DYNAMICS function —
        it says what happens next.
        """
        ...

    async def terminal(self, state: State) -> tuple[bool, dict[str, float] | None]:
        """Check if the situation has reached a natural conclusion.

        Returns (is_terminal, payoffs_or_None).
        """
        ...
```

**Renamed methods**: `get_actions` → `generate`, `transition` → `simulate`, `is_terminal` → `terminal`. The new names are shorter and map directly to the MuZero components they replace: policy → generate, dynamics → simulate, value → evaluate.

**Why only 4 methods**: This is the MINIMUM interface that MCTS needs. Every additional method would be specific to certain scenarios and would pollute the universal interface. Reflection, synthesis, and multi-AI are ABOVE the Oracle level.

### LLM Oracle

The LLMOracle implements the 4-method Oracle protocol using any LLM API (OpenAI, Anthropic, or compatible).

**Key improvements over v1:**

#### 1. Multi-Perspective Evaluation

The evaluation prompt asks the LLM to reason from EACH player's perspective before scoring:

```
Evaluate this strategic position.

SITUATION: {state.description}
PERSPECTIVE (what current player sees): {state.perspective or state.description}
PLAYERS: {players}
GOAL: {state.goal}

For EACH player, briefly consider:
- What they likely want
- What they can observe
- How strong their position is

Then return scores: {"player_name": 0.0-1.0, ...}
```

This produces significantly better evaluations because the LLM explicitly models each player's incentives and information, rather than giving a shallow assessment.

#### 2. Calibrated Priors via Chain-of-Thought

The action generation prompt uses brief chain-of-thought to produce better-calibrated priors:

```
Analyze available actions for {player}.

SITUATION: {description}
GOAL: {goal}

Think about:
- What are the most impactful moves?
- What would an expert strategist consider?
- What might the opponent expect/not expect?

Return 3-5 actions as JSON, with prior reflecting relative promise:
[{"name": "...", "description": "...", "prior": 0.0-1.0}]
```

#### 3. Robust JSON Extraction

Enhanced extraction handles all common LLM output formats:
- Raw JSON
- Markdown-fenced JSON (```json ... ```)
- JSON embedded in explanation text
- Partial/malformed JSON (best-effort recovery)
- Multiple JSON objects (take the first valid one)

#### 4. Synthesis and Reflection (non-Protocol methods)

```python
class LLMOracle:
    # ... Oracle protocol methods ...

    async def synthesize(self, state: State, strategy_data: dict) -> str:
        """Generate natural-language strategic analysis from search results."""
        ...

    async def reflect(self, state: State, strategy_data: dict) -> list[str]:
        """Critique a strategy and identify vulnerabilities.

        Returns list of specific weaknesses/counter-strategies.
        This is NOT part of the Oracle protocol — it's a higher-level
        capability used by the engine's reflection pass.
        """
        ...

    async def decompose_goal(self, state: State) -> list[str]:
        """Break a goal into 2-4 ordered milestones.

        Used for bidirectional reasoning: milestones provide
        backward-derived waypoints that guide forward search.
        """
        ...
```

These are separate from the Protocol because they're engine-level concerns, not search-level.

---

## 5. MCTS Algorithm (`search.py`)

### Overview

The MCTS algorithm runs N simulations, each consisting of 4 phases:
1. **SELECT** — Walk down tree using PUCT, balancing exploitation vs exploration
2. **EXPAND** — Lazily materialize ONE new child node
3. **EVALUATE** — Ask Oracle to score the leaf position
4. **BACKPROPAGATE** — Update all ancestors with the evaluation

### 7 Improvements Over v1

#### Improvement 1: Dirichlet Noise at Root (from AlphaZero)

Before search begins, add random noise to the root's action priors:

```python
prior_new = (1 - ε) * prior_original + ε * Dirichlet(α)
```

Where `ε = 0.25` and `α = 10/num_actions` (AlphaZero defaults).

**Why**: Prevents the search from being completely dominated by the LLM's initial assessment. Even if the LLM assigns low prior to an action, noise gives it a chance to be explored. This catches situations where the LLM's intuition is wrong.

**Cost**: Zero additional LLM calls. Just math on existing priors.

#### Improvement 2: Progressive Widening

Instead of a fixed `max_actions`, the number of expanded children grows with visits:

```python
max_children(N) = ceil(C_pw * N^α)    # C_pw=1.0, α=0.5 by default
```

At 1 visit: 1 child. At 4 visits: 2 children. At 16 visits: 4 children. At 100 visits: 10 children.

**Why**: Early simulations explore the most promising 1-2 actions deeply. As the node accumulates visits, it gradually widens to consider more options. This is optimal allocation of the LLM call budget.

**Implementation**: When `len(node.children) < max_children(node.visits)` and `node._pending` is not empty, expand a new child. Otherwise, select among existing children.

**Cost**: Fewer LLM calls than fixed widening (lazy + progressive = maximally efficient).

#### Improvement 3: Concurrent Simulations with Virtual Loss

Run multiple simulations simultaneously using `asyncio.gather`:

```python
async def search(self, root_state: State) -> Node:
    root = Node(state=root_state)
    self._add_root_noise(root)

    remaining = self.cfg.num_simulations
    while remaining > 0:
        batch = min(self.cfg.parallel_sims, remaining)
        await asyncio.gather(*[self._run_one_sim(root) for _ in range(batch)])
        remaining -= batch
    return root
```

**Virtual loss**: When a simulation selects a node for expansion, it increments `_virtual_loss` on all nodes along the path. This artificially reduces their Q-values, causing concurrent simulations to explore DIFFERENT paths. After evaluation completes, the virtual loss is removed and real values are backpropagated.

```python
def _apply_virtual_loss(self, path: list[Node]):
    for node in path:
        node._virtual_loss += 1
        node.visits += 1  # temporarily count as a visit

def _remove_virtual_loss(self, path: list[Node]):
    for node in path:
        node._virtual_loss -= 1
        node.visits -= 1  # undo temporary visit
```

**Why**: LLM API calls have high latency (~1-3s). Sequential search means 40 simulations × 2s = 80s. With 4-way parallelism: 80s / 4 = 20s. With 8-way: 10s.

**Cost**: Slightly less focused search (virtual loss introduces approximation), but dramatically faster wall-clock time.

#### Improvement 4: Adaptive Exploration Constant

Instead of a fixed `C = 1.41`, adapt based on the value range observed:

```python
C_adaptive = C_base * sqrt(sum_of_squared_values / visits)
```

If the value function has low variance (all states look similar), increase exploration. If high variance (clear differences between actions), reduce exploration to exploit the best path.

**Implementation**: Track running variance of evaluation scores at each node. Adjust C per-node.

**Simpler alternative**: Use the log-scaling from AlphaZero:

```python
C = log((1 + visits + C_base_visits) / C_base_visits) + C_init
```

This naturally increases exploration for nodes with many visits (ensures all children eventually get tried).

#### Improvement 5: Smarter Terminal Detection

Skip expensive `terminal()` calls at shallow depths (already in v1), but also:

- Cache terminal results per state hash
- At deep nodes (near max_depth), treat the node as terminal and use `evaluate()` as the payoff
- If `evaluate()` returns extreme values (>0.95 or <0.05 for any player), heuristically treat as near-terminal

#### Improvement 6: Tree Statistics for Confidence

After search, compute confidence metrics:

```python
def search_confidence(root: Node) -> float:
    """How confident are we in the best action?"""
    if not root.children:
        return 0.0
    visits = sorted([c.visits for c in root.children], reverse=True)
    total = sum(visits)
    if total == 0:
        return 0.0
    top_proportion = visits[0] / total
    uniform = 1.0 / len(visits)
    # Normalize: 0 = uniform (no preference), 1 = complete dominance
    return min((top_proportion - uniform) / (1 - uniform), 1.0)
```

#### Improvement 7: Warm Start (Tree Reuse)

When analyzing a follow-up situation (e.g., opponent made a move), reuse the existing tree:

```python
def warm_start(self, root: Node, action: Action) -> Node:
    """Advance to the child matching this action, preserving the subtree."""
    for child in root.children:
        if child.action and child.action.name == action.name:
            child.parent = None  # detach from old root
            return child
    return None  # no match, start fresh
```

**Why**: In multi-turn scenarios (negotiations, games), the tree from the previous analysis contains valuable information. Reusing it saves ~50% of LLM calls for the next analysis.

### PUCT Selection Formula

```
score(s, a) = Q(s, a) + C · prior(a) · √N(s) / (1 + N(s, a))
```

Where:
- `Q(s, a)` = mean value of action `a` from state `s` (for the current player)
- `C` = exploration constant (default 1.41 ≈ √2)
- `prior(a)` = LLM-estimated promise of action `a`
- `N(s)` = visit count of parent
- `N(s, a)` = visit count of child

This is the PUCT formula from AlphaGo/AlphaZero. It elegantly balances:
- **Exploitation**: high Q-value actions (proven good by simulation)
- **Exploration**: low-visit actions with high priors (promising but untested)
- **Prior-guided**: LLM knowledge steers the search toward expert-level moves

### Lazy Expansion (kept from v1)

The most impactful optimization in the entire codebase:

```
Naive expansion:  oracle.generate() → get 5 actions → oracle.simulate() × 5 → 6 LLM calls
Lazy expansion:   oracle.generate() → get 5 actions → store as _pending
                  → select 1 by prior → oracle.simulate() × 1 → 2 LLM calls
```

For a typical 40-simulation search:
- Naive: ~320 LLM calls
- Lazy: ~80 LLM calls
- **75% reduction in API cost**

The pending actions are stored in `_pending` and materialized one at a time via prior-weighted random selection.

---

## 6. Engine Pipeline (`engine.py`)

### The 4-Phase Pipeline

```python
class Emperor:
    async def analyze(self, situation, *, players, goal, ...) -> Strategy:
        # Phase 1: FRAME — create structured State from natural language
        state = State(description=situation, players=players, goal=goal)

        # Phase 1b: DECOMPOSE (if goal set and milestones enabled)
        if goal and self.cfg.milestones:
            milestones = await self._oracle.decompose_goal(state)
            state.metadata["milestones"] = milestones

        # Phase 2: SEARCH — run MCTS
        root = await self._search.search(state)

        # Phase 3: REFLECT (optional) — critique the strategy
        vulnerabilities = []
        if self.cfg.reflect:
            vulnerabilities = await self._oracle.reflect(state, {
                "best": self._search.best_action(root),
                "scores": self._search.action_scores(root),
                "pv": self._search.principal_variation(root),
            })

        # Phase 4: SYNTHESIZE — generate final analysis
        analysis = await self._oracle.synthesize(state, {
            "best": best, "scores": scores, "pv": pv,
            "vulnerabilities": vulnerabilities,
        })

        return Strategy(
            best_action=best,
            action_scores=scores,
            principal_variation=pv,
            analysis=analysis,
            confidence=self._search.confidence(root),
            vulnerabilities=vulnerabilities,
            stats=self._collect_stats(root, elapsed),
        )
```

### Phase 1: Frame

Convert natural language input into a structured `State`. This is simple — just set the fields. The LLM does the heavy lifting during search.

If `perspective` is provided, it's included in the State for imperfect information handling.

### Phase 1b: Goal Decomposition (Bidirectional Reasoning)

When a goal is provided AND milestones are enabled, the LLM decomposes the goal into 2-4 ordered milestones:

```
Goal: "Get promoted to VP within 2 years"
↓ LLM decomposition
Milestones:
  1. "Deliver a measurable high-impact project result"
  2. "Build strong relationships with executive sponsors"
  3. "Demonstrate strategic vision in leadership forums"
  4. "Secure formal nomination and interview process"
```

These milestones are stored in `state.metadata["milestones"]` and used by the evaluation prompt to score progress toward the goal. This creates **bidirectional reasoning**:

- **Forward**: "What actions move me toward milestone 1?"
- **Backward**: "Milestone 4 requires milestone 3, which requires milestone 2..." (embedded in the milestones themselves)

**Why not explicit backward search?** In natural language, you can't enumerate "states that lead to the goal" or reverse transitions. Goal decomposition achieves the same effect — it provides backward-derived waypoints that guide forward search.

### Phase 2: Search

Run MCTS with all improvements active. This is where 90% of the computation happens.

### Phase 3: Reflect (Optional)

After search completes, have the LLM critique the best strategy:

```
The search engine recommends: {best_action}
Supporting reasoning: {principal_variation}
Action confidence scores: {scores}

Now critically analyze this recommendation:
1. What's the strongest counter-strategy the opponent could use?
2. What assumptions could be wrong?
3. What information are we missing that could change the picture?
4. Under what conditions would this strategy fail?

Return a JSON array of specific vulnerabilities:
["vulnerability 1", "vulnerability 2", ...]
```

**Why this works**: MCTS optimizes for the EXPECTED value of actions. The reflection pass identifies TAIL RISKS — unlikely but devastating outcomes. Together, they cover both the average case and worst case.

**Cost**: 1 additional LLM call. The ROI is enormous — users get not just a recommendation, but a risk assessment.

### Phase 4: Synthesize

Generate the final human-readable analysis, incorporating:
- The search results (best action, scores, variation)
- Vulnerabilities from reflection
- Goal milestones (if applicable)
- Confidence level

The synthesis prompt asks the LLM to produce concise, actionable strategic advice.

---

## 7. Multi-AI Collaboration (`council.py`)

### The Key Insight

The Oracle protocol is the perfect abstraction for multi-AI collaboration. Every multi-AI pattern can be implemented as an **Oracle wrapper** — the MCTS core never changes.

### Pattern 1: Ensemble Oracle

Multiple LLMs evaluate each position. Scores are averaged for robustness.

```python
class EnsembleOracle:
    """Aggregates multiple oracles for robust evaluation."""

    def __init__(self, oracles: list[Oracle]):
        self.oracles = oracles
        self._primary = oracles[0]  # one oracle handles generation/simulation

    async def generate(self, state: State) -> list[Action]:
        return await self._primary.generate(state)  # one oracle generates

    async def evaluate(self, state: State) -> dict[str, float]:
        # ALL oracles evaluate in parallel, average scores
        results = await asyncio.gather(*[o.evaluate(state) for o in self.oracles])
        all_players = set()
        for r in results:
            all_players.update(r.keys())
        return {
            p: sum(r.get(p, 0.5) for r in results) / len(results)
            for p in all_players
        }

    async def simulate(self, state: State, action: Action) -> State:
        return await self._primary.simulate(state, action)  # one oracle simulates

    async def terminal(self, state: State) -> tuple[bool, dict[str, float] | None]:
        return await self._primary.terminal(state)
```

**Why ensemble?** LLM evaluations have variance. One call might rate a position at 0.7, another at 0.4. Averaging N evaluations reduces variance by √N. With 3 oracles, we get ~42% lower variance at 3× evaluation cost.

**When to use**: High-stakes decisions where accuracy matters more than cost.

**How to create**: Different temperature settings, different models, or same model with different prompts.

```python
oracles = [
    LLMOracle(LLMConfig(temperature=0.3)),  # conservative
    LLMOracle(LLMConfig(temperature=0.7)),  # balanced
    LLMOracle(LLMConfig(temperature=1.0)),  # creative
]
council = EnsembleOracle(oracles)
mcts = MCTS(council, config)
```

### Pattern 2: Adversarial Oracle (Self-Play)

One AI proposes the strategy. Another AI plays as the opponent, trying to exploit it.

```python
class AdversarialOracle:
    """Wraps an oracle with adversarial testing.

    During evaluation, considers the opponent's best response.
    """

    def __init__(self, base: Oracle, adversary: Oracle | None = None):
        self.base = base
        self.adversary = adversary or base  # can use same oracle

    async def evaluate(self, state: State) -> dict[str, float]:
        # Base evaluation
        base_scores = await self.base.evaluate(state)

        # Adversary's counter-evaluation: what if opponent plays optimally?
        opponent_state = State(
            description=state.description,
            players=state.players,
            current_player=(state.current_player + 1) % len(state.players),
            history=state.history,
            goal=f"Counter the strategy of {state.whose_turn}",
        )
        counter_scores = await self.adversary.evaluate(opponent_state)

        # Blend: pessimistic adjustment
        # If the adversary thinks our position is weaker, believe them partially
        blended = {}
        for p in set(list(base_scores.keys()) + list(counter_scores.keys())):
            b = base_scores.get(p, 0.5)
            c = counter_scores.get(p, 0.5)
            blended[p] = 0.7 * b + 0.3 * c  # lean toward base, temper with adversary
        return blended
```

**Why adversarial?** Standard MCTS assumes the ORACLE's evaluation is correct. But the LLM might be optimistic about our position. The adversarial oracle provides a "second opinion" from the opponent's perspective, catching blind spots.

### Pattern 3: Supervised Oracle

One AI generates, another validates. Catches hallucinations and unrealistic actions.

```python
class SupervisedOracle:
    """Worker oracle generates, supervisor validates."""

    def __init__(self, worker: Oracle, supervisor: Oracle):
        self.worker = worker
        self.supervisor = supervisor

    async def generate(self, state: State) -> list[Action]:
        actions = await self.worker.generate(state)
        # Supervisor re-evaluates priors
        # (validate that actions are realistic and well-calibrated)
        validated = await self.supervisor.evaluate_actions(state, actions)
        return validated

    async def evaluate(self, state: State) -> dict[str, float]:
        # Both evaluate, supervisor has final say on disagreements
        w = await self.worker.evaluate(state)
        s = await self.supervisor.evaluate(state)
        # If they agree (within 0.15), use worker's result (cheaper)
        # If they disagree, use supervisor's (more careful)
        result = {}
        for p in set(list(w.keys()) + list(s.keys())):
            wv, sv = w.get(p, 0.5), s.get(p, 0.5)
            result[p] = wv if abs(wv - sv) < 0.15 else sv
        return result
```

### Composability

All Oracle wrappers can be composed:

```python
# Create base oracles
gpt = LLMOracle(LLMConfig(provider="openai", model="gpt-4o"))
claude = LLMOracle(LLMConfig(provider="anthropic"))

# Ensemble of GPT + Claude
ensemble = EnsembleOracle([gpt, claude])

# Add adversarial testing on top
robust = AdversarialOracle(ensemble)

# Use with MCTS — no changes to search algorithm
mcts = MCTS(robust, MCTSConfig(num_simulations=60))
```

### Multi-AI Collaboration: Addressing Hallucination and Drift

The user correctly identified key risks in multi-AI systems:

| Risk | Mitigation |
|------|------------|
| **Hallucination** | Ensemble averaging reduces individual hallucinations; supervised mode validates |
| **Goal drift** | Every Oracle call includes the original goal in the prompt; State is immutable |
| **Context loss** | Full State (description + history + goal) is passed to EVERY call |
| **Inconsistency** | One primary oracle handles generation/simulation; ensemble only for evaluation |
| **Compounding errors** | Reflection pass at the END catches accumulated errors |

**Critical design decision**: Only EVALUATION is ensembled. Generation (actions) and simulation (transitions) use a single oracle. This prevents inconsistency — the world model must be coherent, even if the value judgments benefit from multiple perspectives.

---

## 8. Bidirectional Reasoning

### The Problem

Traditional MCTS searches FORWARD from the current state. But strategic planning often requires BACKWARD reasoning from the goal. "What needs to happen for me to achieve X?"

### The Solution: Goal-Conditioned MCTS with Milestone Decomposition

Emperor implements bidirectional reasoning through three mechanisms:

#### Mechanism 1: Goal in State

The `goal` field is included in every Oracle prompt. The LLM naturally incorporates backward reasoning when it knows the goal:

```python
# Without goal: LLM evaluates "is this position good?"
# With goal:    LLM evaluates "does this position advance toward the goal?"
```

This is implicit bidirectional reasoning — the LLM has been trained on planning and reasoning backward from goals.

#### Mechanism 2: Milestone Decomposition (Explicit Backward Planning)

When enabled, the engine asks the LLM to decompose the goal into milestones BEFORE search:

```
Goal: "Close a $2M Series A round"

Milestones (LLM-generated):
1. "Build a compelling pitch deck with traction metrics"
2. "Secure 3-5 warm introductions to target investors"
3. "Complete 10+ investor meetings and generate term sheets"
4. "Negotiate terms and close lead investor commitment"
```

These milestones are then embedded in the evaluation prompt:

```
Evaluate this position. Consider progress toward these milestones:
1. {milestone_1}
2. {milestone_2}
...

Score 0.0-1.0 for each player, where 1.0 = on track to achieve all milestones.
```

This creates explicit backward reasoning: milestones are derived from the goal (backward), and the search evaluates progress toward them (forward).

#### Mechanism 3: Backward-Aware Action Generation

When milestones exist, the action generation prompt includes them:

```
Given the current situation and these milestones to achieve:
{milestones}

What actions most directly advance toward the NEXT uncompleted milestone?
```

This focuses the search on actions that bridge the gap between current state and goal-derived milestones.

### Why Not Two Separate Trees?

Explicit bidirectional tree search (forward tree + backward tree + connection detection) was considered and rejected:

1. **State matching is impossible in natural language** — when does a forward state "connect" to a backward state? Approximate string matching is unreliable.
2. **Backward transitions don't exist** — you can't reliably ask "what state LEADS to this?" for arbitrary natural language states.
3. **The LLM already reasons bidirectionally** — when given a goal, it naturally considers both "what to do next" and "what's needed for the goal."
4. **Milestone decomposition achieves the same result** — backward-derived waypoints guide forward search, without the complexity of maintaining two trees.

---

## 9. Configuration (`EmperorConfig`)

```python
@dataclass
class EmperorConfig:
    # ── LLM ──────────────────────────────────────────────
    provider: str = "openai"           # "openai" | "anthropic"
    model: str = ""                    # Auto-detected if empty
    api_key: str = ""                  # From env if empty
    temperature: float = 0.7

    # ── Search ───────────────────────────────────────────
    simulations: int = 40              # MCTS iterations
    exploration: float = 1.41          # PUCT exploration constant (≈√2)
    max_depth: int = 8                 # Maximum tree depth
    max_actions: int = 7               # Max actions per node (with progressive widening)
    parallel_sims: int = 1             # Concurrent simulations (1 = sequential)

    # ── Progressive Widening ─────────────────────────────
    progressive_widening: bool = True  # Dynamic action expansion
    pw_alpha: float = 0.5              # Widening exponent
    pw_c: float = 1.5                  # Widening coefficient

    # ── Exploration Noise ────────────────────────────────
    root_noise: bool = True            # Dirichlet noise at root
    noise_epsilon: float = 0.25        # Noise mixing ratio
    noise_alpha: float = 0.3           # Dirichlet concentration (auto: 10/num_actions)

    # ── Optional Features ────────────────────────────────
    reflect: bool = False              # Enable reflection pass
    milestones: bool = False           # Enable goal decomposition
    ensemble_size: int = 1             # >1 enables ensemble evaluation

    # ── Performance ──────────────────────────────────────
    cache: bool = True                 # Cache LLM responses
    terminal_check_depth: int = 2      # Skip terminal checks at shallow depths
```

### Configuration Profiles

```python
# Quick analysis (fast, cheap)
QUICK = EmperorConfig(simulations=20, max_depth=5, parallel_sims=4)

# Standard analysis (balanced)
STANDARD = EmperorConfig(simulations=40, max_depth=8, reflect=True)

# Deep analysis (thorough, expensive)
DEEP = EmperorConfig(
    simulations=100, max_depth=12, reflect=True,
    milestones=True, ensemble_size=3, parallel_sims=8,
)

# Formal game (no LLM, fast)
GAME = EmperorConfig(simulations=1000, max_depth=20, max_actions=20)
```

---

## 10. API Design

### Python API

```python
import asyncio
from emperor import Emperor, EmperorConfig

async def main():
    # Simple usage
    async with Emperor() as e:
        strategy = await e.analyze(
            "My competitor just dropped prices by 20%. Our margins are thin.",
            players=["Our Company", "Competitor", "Customers"],
            goal="Maintain market share while protecting margins",
        )
        print(strategy.best_action.name)
        print(strategy.analysis)
        print(f"Confidence: {strategy.confidence:.0%}")
        if strategy.vulnerabilities:
            print("Watch out for:", strategy.vulnerabilities)

    # Advanced: multi-AI with reflection
    async with Emperor(EmperorConfig(
        provider="anthropic",
        simulations=60,
        reflect=True,
        milestones=True,
        ensemble_size=2,
        parallel_sims=4,
    )) as e:
        strategy = await e.analyze(
            "We have 3 term sheets. Lead investor wants 2x liquidation preference.",
            players=["Founder", "Lead Investor", "Other Investors"],
            goal="Close round with founder-friendly terms",
        )

asyncio.run(main())
```

### Formal Game API

```python
from emperor import GameOracle, MCTS, MCTSConfig, Action, State

class Chess(GameOracle):
    def legal_actions(self, state): ...
    def apply_action(self, state, action): ...
    def score(self, state): ...
    def terminal(self, state): ...

game = Chess()
mcts = MCTS(game, MCTSConfig(num_simulations=10000, parallel_sims=8))
root = await mcts.search(initial_state)
```

### CLI

```bash
# One-shot analysis
emperor "Should we accept the acquisition at 5x ARR?" \
  --goal "Maximize long-term value" \
  --players "Founder,Acquirer,Board"

# Deep analysis with reflection
emperor "Our key engineer is being recruited by Google" \
  --goal "Retain the engineer without overspending" \
  --reflect --milestones \
  --simulations 60

# Interactive mode
emperor --interactive

# Formal game
emperor --game tictactoe --simulations 1000
```


