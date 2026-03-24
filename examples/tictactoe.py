"""TicTacToe — Strategist solves a perfect-information game via pure MCTS.

This proves the engine is truly universal:
  - No LLM calls needed
  - GameOracle provides exact rules
  - MCTS converges to optimal play (minimax)

Run: python examples/tictactoe.py
"""

import asyncio
from strategist import MCTS, MCTSConfig, Action, State, text_tree
from strategist.game import GameOracle


class TicTacToe(GameOracle):
    """Tic-tac-toe rules as a GameOracle."""

    def _board(self, state: State) -> list[str]:
        """Extract 3x3 board from state description. '.' = empty."""
        lines = state.description.strip().split("\n")
        return [ch for line in lines for ch in line.split() if ch in ".XO"]

    def _to_desc(self, board: list[str]) -> str:
        return "\n".join(
            " ".join(board[i * 3 : i * 3 + 3]) for i in range(3)
        )

    def _mark(self, state: State) -> str:
        return "X" if state.current_player == 0 else "O"

    def _winner(self, board: list[str]) -> str | None:
        wins = [
            (0,1,2),(3,4,5),(6,7,8),  # rows
            (0,3,6),(1,4,7),(2,5,8),  # cols
            (0,4,8),(2,4,6),          # diags
        ]
        for a, b, c in wins:
            if board[a] == board[b] == board[c] != ".":
                return board[a]
        return None

    def legal_actions(self, state: State) -> list[Action]:
        board = self._board(state)
        mark = self._mark(state)
        actions = []
        for i, cell in enumerate(board):
            if cell == ".":
                r, c = divmod(i, 3)
                actions.append(Action(
                    name=f"{mark}→({r},{c})",
                    description=f"Place {mark} at row {r}, col {c}",
                ))
        return actions

    def apply_action(self, state: State, action: Action) -> State:
        board = self._board(state)
        mark = self._mark(state)
        # Parse position from action name: "X→(r,c)"
        pos_str = action.name.split("→")[1].strip("()")
        r, c = int(pos_str.split(",")[0]), int(pos_str.split(",")[1])
        board[r * 3 + c] = mark
        return state.advance(
            description=self._to_desc(board),
            action_desc=action.name,
        )

    def terminal(self, state: State) -> bool:
        board = self._board(state)
        return self._winner(board) is not None or "." not in board

    def score(self, state: State) -> dict[str, float]:
        board = self._board(state)
        w = self._winner(board)
        if w == "X":
            return {"X": 1.0, "O": 0.0}
        elif w == "O":
            return {"X": 0.0, "O": 1.0}
        return {"X": 0.5, "O": 0.5}  # draw


async def main():
    game = TicTacToe()
    mcts = MCTS(game, MCTSConfig(num_simulations=500, max_depth=9, terminal_check_depth=0))

    # Start from empty board
    state = State(
        description=". . .\n. . .\n. . .",
        players=["X", "O"],
    )

    print("\n  Tic-Tac-Toe — MCTS vs MCTS\n")

    while True:
        # Check terminal
        if game.terminal(state):
            board = game._board(state)
            w = game._winner(board)
            print(f"\n  {'Draw!' if not w else f'{w} wins!'}\n")
            break

        # Search
        root = await mcts.search(state)
        best = mcts.best_action(root)

        # Show tree for first move
        if not state.history:
            print("  Search tree (first move):")
            print(text_tree(root, max_depth=2, min_visits=10))
            print()

        # Apply
        state = await game.transition(state, best)
        print(f"  {best.name}")
        print(f"  {state.description.replace(chr(10), '  |  ')}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
