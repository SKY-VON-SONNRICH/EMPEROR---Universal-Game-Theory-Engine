# Emperor

**Universal Strategic Reasoning Engine — MCTS × LLM = Game Theory for Everything**

Feed any real-world strategic dilemma in plain English. Get back a game-theoretic analysis that would cost $2,000 from McKinsey

> Describe any strategic scenario in natural language. Get rigorous game-theoretic analysis back.

Emperor combines Monte Carlo Tree Search with LLM intelligence to create a **domain-agnostic strategy engine**. It works for business decisions, negotiations, game strategy, interpersonal dynamics, investment, career planning, or any scenario where multiple agents make strategic choices.


**~1,200 lines of core code. 1 dependency (`httpx`). Any domain. No training.**

## Architecture

```
User (natural language)
        │
        ▼
  ┌───────────┐
  │  Emperor   │
  │            │
  │ 1. Frame   │  Parse situation → structured State
  │ 2. Search  │  MCTS + PUCT with LLM Oracle
  │ 3. Reflect │  (optional) Critique strategy, find weaknesses
  │ 4. Synth   │  Generate final analysis + recommendation
  └─────┬──────┘
        │
  ┌─────┴──────────────────────────────┐
  │           Oracle (4 methods)        │
  │  generate | evaluate | simulate |   │
  │  terminal                           │
  │                                     │
  │  Implementations:                   │
  │  • LLMOracle  (GPT / Claude)       │
  │  • GameOracle (coded rules)        │
  │  • EnsembleOracle (multi-AI)       │
  │  • AdversarialOracle (self-play)   │
  └─────────────────────────────────────┘
```

## Install & Run

```bash
# Backend
pip install fastapi uvicorn websockets httpx

# Frontend
cd web && npm install && npm run build && cd ..

# Start Emperor (serves UI on http://localhost:8000)
python -m emperor
```

For development (hot-reload):
```bash
# Terminal 1: Backend
uvicorn emperor.server:app --reload --port 8000

# Terminal 2: Frontend
cd web && npm run dev
# Opens at http://localhost:5173 (proxied to backend)
```

## Web UI

3-panel interface:
- **Left**: Session list — create, switch, delete sessions
- **Center**: Chat — Send (chat) or Analyze (MCTS search) your strategic questions
- **Right**: D3.js tree visualization — interactive, zoomable search tree

Set your **Anthropic or OpenAI API key** in Settings (gear icon). Keys are stored in your browser only.

## Quick Start (Python API)

```python
import asyncio
from emperor import Emperor

async def main():
    async with Emperor() as e:
        strategy = await e.analyze(
            "I'm negotiating a salary raise from $120K. "
            "Market rate is $150K. I have a competing offer at $155K.",
            players=["You", "Manager"],
            goal="Maximize salary while preserving relationship",
        )
        print(f"Best move: {strategy.best_action.name}")
        print(f"Confidence: {strategy.confidence:.0%}")
        print(f"Analysis: {strategy.analysis}")
        if strategy.vulnerabilities:
            print(f"Watch out: {strategy.vulnerabilities}")

asyncio.run(main())
```

## Configuration

```python
from emperor import Emperor, EmperorConfig

engine = Emperor(EmperorConfig(
    # LLM
    provider="anthropic",              # "openai" | "anthropic"
    model="claude-sonnet-4-20250514",  # auto-detected if empty
    temperature=0.7,

    # Search
    simulations=40,                    # MCTS iterations (more = deeper)
    exploration=1.41,                  # PUCT exploration constant
    max_depth=8,                       # max lookahead depth
    max_actions=7,                     # max actions per node
    parallel_sims=4,                   # concurrent simulations

    # Intelligence (optional, all OFF by default)
    reflect=True,                      # LLM critiques strategy after search
    milestones=True,                   # goal decomposition for backward reasoning
    ensemble_size=1,                   # >1 enables multi-AI ensemble evaluation
))
```


## How It Works

### MCTS + PUCT Selection

Each simulation:
1. **Select** — walk down tree using PUCT, balancing exploration vs exploitation weighted by LLM priors
2. **Expand** — **lazily** materialize ONE new child (cuts LLM calls by ~75%)
3. **Evaluate** — LLM scores the position for each player (multi-perspective)
4. **Backpropagate** — update all ancestor nodes with the evaluation

Selection formula (PUCT from AlphaGo/AlphaZero):
```
score(s,a) = Q(s,a) + C · prior(a) · √N(s) / (1 + N(s,a))
```

### Key Improvements (v2)

| Feature | How | Impact |
|---------|-----|--------|
| **Dirichlet noise** | Random noise on root priors (from AlphaZero) | Prevents local optima, finds surprising strategies |
| **Progressive widening** | `k = C·N^0.5` actions per node | Optimal LLM call budget allocation |
| **Concurrent search** | Virtual loss + asyncio.gather | 4-8x faster wall clock time |
| **Multi-perspective eval** | LLM reasons from each player's POV | Much richer strategic understanding |
| **Reflection pass** | LLM critiques strategy post-search | Identifies tail risks and counter-strategies |
| **Goal decomposition** | LLM breaks goal into milestones | Bidirectional reasoning without dual trees |
| **Confidence scoring** | Derived from visit distribution | Tells you how certain the recommendation is |

### Bidirectional Reasoning

When you provide a `goal`, Emperor provides bidirectional search:

- **Forward**: MCTS explores future possibilities from current state
- **Backward**: Goal is decomposed into milestones, evaluation scores progress toward them

```python
strategy = await engine.analyze(
    "My startup has $500K ARR and 3 enterprise leads.",
    goal="Reach $2M ARR within 18 months",  # enables backward reasoning
)
# The engine will:
# 1. Decompose goal into milestones (e.g., "close 2 enterprise deals", "expand existing accounts")
# 2. Search for actions that advance toward milestones
# 3. Evaluate progress at each tree node
```

### Imperfect Information

For scenarios with hidden information (e.g., poker, negotiation with hidden agendas):

```python
strategy = await engine.analyze(
    "salary negotiation, I know market rate is $150K",
    players=["You", "HR Manager"],
    goal="Get at least $145K",
    perspective="You don't know the company's budget ceiling or competing candidates",
)
```

The `perspective` field tells the Oracle what the current player can see, enabling information-asymmetric reasoning.

### Multi-Player Support

Emperor handles any number of players. Each player maximizes their own utility:

- **Zero-sum** (chess, poker) — one player's gain is another's loss
- **General-sum** (negotiations) — win-win or lose-lose outcomes exist
- **Cooperative** (team strategy) — aligned incentives
- **Mixed** (business) — cooperation + competition simultaneously

## Multi-AI Collaboration

Emperor supports composable multi-AI patterns through **Oracle wrappers**. The MCTS core never changes.

### Ensemble (robust evaluation)

Multiple LLMs evaluate each position. Scores are averaged for lower variance:

```python
from emperor import Emperor, EmperorConfig

# 3-model ensemble: evaluations are averaged
engine = Emperor(EmperorConfig(ensemble_size=3))
```

### Adversarial (self-play testing)

One AI proposes strategy, another plays as opponent:

```python
from emperor import AdversarialOracle, LLMOracle, MCTS

base = LLMOracle()
robust = AdversarialOracle(base)  # adds opponent's perspective to evaluation
mcts = MCTS(robust)
```

### Supervised (hallucination prevention)

Worker generates, supervisor validates:

```python
from emperor import SupervisedOracle, LLMOracle

worker = LLMOracle(LLMConfig(model="gpt-4o-mini"))       # fast, cheap
supervisor = LLMOracle(LLMConfig(model="gpt-4o"))         # careful, accurate
oracle = SupervisedOracle(worker, supervisor)
```

### Compose Freely

All wrappers implement the Oracle protocol and can be combined:

```python
gpt = LLMOracle(LLMConfig(provider="openai"))
claude = LLMOracle(LLMConfig(provider="anthropic"))
ensemble = EnsembleOracle([gpt, claude])          # multi-model ensemble
robust = AdversarialOracle(ensemble)               # + adversarial testing
# Use with MCTS — zero code changes to search algorithm
```

## Formal Games (no LLM)

For games with explicit rules, extend `GameOracle`:

```python
from emperor import GameOracle, Action, State

class TicTacToe(GameOracle):
    def legal_actions(self, state) -> list[Action]: ...
    def apply_action(self, state, action) -> State: ...
    def score(self, state) -> dict[str, float]: ...
    def terminal(self, state) -> bool: ...
```

Search with pure MCTS (no API calls, millisecond speed):

```python
from emperor import MCTS, MCTSConfig

game = TicTacToe()
mcts = MCTS(game, MCTSConfig(num_simulations=1000))
root = await mcts.search(state)
```

## Tree Visualization

```python
from emperor import text_tree, json_tree, mermaid_tree

# ASCII tree (terminal)
print(text_tree(root, max_depth=3))
# root [n=40 You:62%, Opponent:38%]
# ├── counter_offer [n=18 You:68%, Opponent:32%]
# │   ├── accept [n=8 You:72%, Opponent:28%]
# │   └── negotiate [n=6 You:65%, Opponent:35%]
# └── accept_current [n=12 You:45%, Opponent:55%]

# JSON (for web UIs)
data = json_tree(root)

# Mermaid (for GitHub/docs)
print(mermaid_tree(root))
```

## Custom Oracle

Implement 4 async methods to use any intelligence source:

```python
class MyOracle:
    async def generate(self, state: State) -> list[Action]: ...
    async def evaluate(self, state: State) -> dict[str, float]: ...
    async def simulate(self, state: State, action: Action) -> State: ...
    async def terminal(self, state: State) -> tuple[bool, dict[str, float] | None]: ...

mcts = MCTS(MyOracle(), MCTSConfig(num_simulations=100))
root = await mcts.search(state)
```

## Example Scenarios

| Domain | Scenario |
|--------|----------|
| **Business** | Pricing strategy, market entry, M&A, competitive response |
| **Negotiation** | Salary, contracts, diplomatic disputes, vendor deals |
| **Game Strategy** | Chess positions, poker decisions, board game tactics |
| **Interpersonal** | Conflict resolution, team dynamics, persuasion, influence |
| **Investment** | Portfolio allocation, venture decisions, risk management |
| **Career** | Job offers, project selection, skill development paths |
| **Geopolitics** | Alliance formation, trade negotiations, conflict de-escalation |

## Design Document

See [DESIGN.md](./DESIGN.md) for the complete technical architecture

## License

MIT
