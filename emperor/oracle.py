"""LLM Oracle — the universal world model.

The Oracle is the sole coupling point between MCTS and domain knowledge.
An LLM serves quadruple duty as:
  1. Action generator  (generate)
  2. State evaluator   (evaluate)
  3. World simulator   (simulate)
  4. Terminal detector  (terminal)

This collapses AlphaZero's rules + policy + value networks into a single LLM.
"""

from __future__ import annotations

import asyncio
import json
import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx

from .core import Action, State


# ══════════════════════════════════════════════════════════
#  Oracle Protocol — implement this for any intelligence source
# ══════════════════════════════════════════════════════════

@runtime_checkable
class Oracle(Protocol):
    async def generate(self, state: State) -> list[Action]: ...
    async def evaluate(self, state: State) -> dict[str, float]: ...
    async def simulate(self, state: State, action: Action) -> State: ...
    async def terminal(self, state: State) -> tuple[bool, dict[str, float] | None]: ...


# ══════════════════════════════════════════════════════════
#  LLM Oracle — works with OpenAI, Anthropic, or any compatible API
# ══════════════════════════════════════════════════════════

@dataclass
class LLMConfig:
    provider: str = "anthropic"       # "openai" | "anthropic"
    model: str = ""                   # auto-detected if empty
    api_key: str = ""                 # from env if empty
    temperature: float = 0.7
    max_tokens: int = 2048
    cache_enabled: bool = True
    reasoning: str = "standard"       # "standard" | "extended"

    def __post_init__(self):
        if not self.model:
            self.model = {
                "openai": "gpt-5.1",
                "anthropic": "claude-opus-4-20250514",
            }.get(self.provider, "gpt-5.1")
        if not self.api_key:
            env_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
            self.api_key = os.getenv(env_map.get(self.provider, ""), "")

    @property
    def is_extended(self) -> bool:
        """Anthropic extended thinking or OpenAI pro mode."""
        return self.reasoning in ("extended", "pro")

    @property
    def is_openai_reasoning(self) -> bool:
        """GPT-5 and o-series use reasoning API params."""
        return self.provider == "openai" and (
            self.model.startswith("o") or self.model.startswith("gpt-5")
        )


SYSTEM_PROMPT = """\
You are the Oracle of a strategic decision engine. You analyze strategic \
situations with game-theoretic rigor and return ONLY valid JSON. \
No markdown fences, no explanation outside JSON.

Rules:
- Be realistic and precise in your analysis.
- Consider all players' incentives, constraints, and likely behaviors.
- Account for uncertainty and hidden information.
- Probabilities must be calibrated and honest.\
"""


class LLMOracle:
    """Universal Oracle backed by any LLM."""

    def __init__(self, config: LLMConfig | None = None):
        self.cfg = config or LLMConfig()
        self._client = httpx.AsyncClient(timeout=60.0)
        self._cache: dict[str, str] = {}
        self.call_count = 0

    # ── Public Oracle interface ───────────────────────────

    async def generate(self, state: State) -> list[Action]:
        goal_line = f"\nGOAL: {state.goal}" if state.goal else ""
        history_line = "\n".join(state.history[-6:]) if state.history else "None yet"

        prompt = (
            f"Analyze the available actions for {state.whose_turn}.\n\n"
            f"SITUATION:\n{state.description}\n\n"
            f"PLAYERS: {', '.join(state.players)}\n"
            f"RECENT HISTORY:\n{history_line}"
            f"{goal_line}\n\n"
            f"Return a JSON array of 3-5 actions sorted by promise:\n"
            f'[{{"name":"short_name","description":"what this involves","prior":0.0-1.0}}]\n\n'
            f"Priors should roughly sum to 1.0. Higher = more promising."
        )

        raw = await self._call(prompt)
        try:
            items = json.loads(self._extract_json(raw))
            total = sum(max(it.get("prior", 0.2), 0.01) for it in items) or 1.0
            return [
                Action(
                    name=it["name"],
                    description=it.get("description", ""),
                    prior=max(it.get("prior", 0.2), 0.01) / total,
                )
                for it in items
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            return [Action(name="default", description="Continue current approach", prior=1.0)]

    async def evaluate(self, state: State) -> dict[str, float]:
        goal_line = f"\nGOAL for evaluation context: {state.goal}" if state.goal else ""
        perspective_line = f"\nPERSPECTIVE: {state.perspective}" if state.perspective else ""

        prompt = (
            f"Evaluate this strategic position for each player.\n"
            f"Score 0.0 = worst possible, 0.5 = neutral, 1.0 = best possible.\n\n"
            f"SITUATION:\n{state.description}\n\n"
            f"PLAYERS: {', '.join(state.players)}\n"
            f"HISTORY:\n{chr(10).join(state.history[-4:]) or 'None yet'}"
            f"{goal_line}{perspective_line}\n\n"
            f'Return JSON: {{"player_name": score, ...}}'
        )

        raw = await self._call(prompt)
        try:
            result = json.loads(self._extract_json(raw))
            return {str(k): float(v) for k, v in result.items()}
        except (json.JSONDecodeError, ValueError, TypeError):
            return {p: 0.5 for p in state.players}

    async def simulate(self, state: State, action: Action) -> State:
        prompt = (
            f"Simulate the outcome of this action. Describe the new situation.\n\n"
            f"CURRENT SITUATION:\n{state.description}\n\n"
            f"ACTION by {state.whose_turn}: {action.description or action.name}\n\n"
            f"Return JSON:\n"
            f'{{"description":"new situation after the action"}}'
        )

        raw = await self._call(prompt)
        try:
            result = json.loads(self._extract_json(raw))
            return state.advance(
                description=result["description"],
                action_desc=action.description or action.name,
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return state.advance(
                description=f"{state.description} → {state.whose_turn} chose: {action.name}",
                action_desc=action.name,
            )

    async def terminal(self, state: State) -> tuple[bool, dict[str, float] | None]:
        prompt = (
            f"Is this situation at a natural conclusion or decision endpoint? "
            f"Respond ONLY with JSON.\n\n"
            f"SITUATION:\n{state.description}\n\n"
            f"HISTORY ({len(state.history)} moves):\n"
            f"{chr(10).join(state.history[-4:]) or 'None yet'}\n\n"
            f'Return: {{"terminal": true/false, "payoffs": {{"player": 0.0-1.0}} or null}}'
        )
        raw = await self._call(prompt)
        try:
            result = json.loads(self._extract_json(raw))
            if result.get("terminal"):
                payoffs = result.get("payoffs") or {p: 0.5 for p in state.players}
                return True, {str(k): float(v) for k, v in payoffs.items()}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return False, None

    # ── Synthesis & Reflection (engine-level, not search) ─

    async def synthesize(self, state: State, pv: list[Action], scores: list[tuple[Action, float]]) -> str:
        scores_text = "\n".join(f"  {a.name} ({a.description}): {s:.0%}" for a, s in scores)
        pv_text = " → ".join(a.name for a in pv) if pv else "N/A"

        prompt = (
            f"You are a strategic advisor. Provide a concise analysis.\n\n"
            f"SITUATION:\n{state.description}\n"
            f"GOAL: {state.goal or 'Maximize advantage'}\n\n"
            f"SEARCH RESULTS:\n"
            f"Action confidence:\n{scores_text}\n\n"
            f"Best play sequence: {pv_text}\n\n"
            f"Provide a 3-5 sentence strategic analysis explaining:\n"
            f"1. Why the top action is strongest\n"
            f"2. Key risks to watch\n"
            f"3. What to do if the opponent deviates\n\n"
            f"Return plain text (not JSON)."
        )
        return await self._call(prompt, is_json=False)

    async def reflect(self, state: State, best_action: Action, scores: list[tuple[Action, float]]) -> list[str]:
        scores_text = "\n".join(f"  {a.name}: {s:.0%}" for a, s in scores[:3])
        prompt = (
            f"Critically analyze this strategic recommendation.\n\n"
            f"SITUATION:\n{state.description}\n"
            f"RECOMMENDED: {best_action.name} — {best_action.description}\n"
            f"ALTERNATIVES:\n{scores_text}\n\n"
            f"Identify 2-3 specific vulnerabilities or counter-strategies:\n"
            f'Return JSON: ["vulnerability 1", "vulnerability 2", ...]'
        )
        raw = await self._call(prompt)
        try:
            return json.loads(self._extract_json(raw))
        except (json.JSONDecodeError, TypeError):
            return []

    async def chat(self, messages: list[dict], system: str = "") -> str:
        """Direct chat — not part of search, used for conversational responses."""
        sys = system or "You are Emperor, a strategic reasoning AI. Answer concisely."
        if self.cfg.provider == "anthropic":
            body: dict = {
                "model": self.cfg.model,
                "system": sys,
                "messages": messages,
            }
            if self.cfg.is_extended:
                body["temperature"] = 1
                body["max_tokens"] = 16000
                body["thinking"] = {"type": "enabled", "budget_tokens": 10000}
            else:
                body["temperature"] = self.cfg.temperature
                body["max_tokens"] = self.cfg.max_tokens

            resp = await self._client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.cfg.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            content = resp.json()["content"]
            for block in content:
                if block.get("type") == "text":
                    return block["text"]
            return content[-1].get("text", "")
        else:
            body: dict = {
                "model": self.cfg.model,
                "messages": [{"role": "system", "content": sys}, *messages],
            }
            if self.cfg.is_openai_reasoning:
                effort = "high" if self.cfg.reasoning == "pro" else "medium"
                body["reasoning_effort"] = effort
                body["max_completion_tokens"] = self.cfg.max_tokens * (4 if self.cfg.reasoning == "pro" else 2)
            else:
                body["temperature"] = self.cfg.temperature
                body["max_tokens"] = self.cfg.max_tokens

            resp = await self._client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.cfg.api_key}"},
                json=body,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    # ── LLM communication layer ───────────────────────────

    async def _call(self, user_prompt: str, is_json: bool = True) -> str:
        cache_key = hashlib.md5(f"{self.cfg.model}:{user_prompt}".encode()).hexdigest()
        if self.cfg.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        self.call_count += 1
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                if self.cfg.provider == "anthropic":
                    result = await self._call_anthropic(user_prompt)
                else:
                    result = await self._call_openai(user_prompt)
                if self.cfg.cache_enabled:
                    self._cache[cache_key] = result
                return result
            except (httpx.HTTPError, httpx.TimeoutException, KeyError) as e:
                last_err = e
                if attempt < 2:
                    await asyncio.sleep(1.5 ** attempt)

        logging.warning("LLM call failed after 3 retries: %s", last_err)
        return "{}"

    async def _call_openai(self, prompt: str) -> str:
        body: dict = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        if self.cfg.is_openai_reasoning:
            # GPT-5 / o-series: reasoning_effort maps thinking→medium, pro→high
            effort = "high" if self.cfg.reasoning == "pro" else "medium"
            body["reasoning_effort"] = effort
            body["max_completion_tokens"] = self.cfg.max_tokens * (4 if self.cfg.reasoning == "pro" else 2)
        else:
            body["temperature"] = self.cfg.temperature
            body["max_tokens"] = self.cfg.max_tokens

        resp = await self._client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.cfg.api_key}"},
            json=body,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def _call_anthropic(self, prompt: str) -> str:
        body: dict = {
            "model": self.cfg.model,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }

        if self.cfg.is_extended:
            # Extended thinking: use thinking block, budget tokens, temperature=1
            body["temperature"] = 1  # required for extended thinking
            body["max_tokens"] = 16000
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": 10000,
            }
        else:
            body["temperature"] = self.cfg.temperature
            body["max_tokens"] = self.cfg.max_tokens

        resp = await self._client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.cfg.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
        )
        resp.raise_for_status()
        # Extract text from response (may contain thinking blocks + text blocks)
        content = resp.json()["content"]
        for block in content:
            if block.get("type") == "text":
                return block["text"]
        return content[-1].get("text", "{}")

    @staticmethod
    def _extract_json(raw: str) -> str:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        for i, ch in enumerate(text):
            if ch in "[{":
                depth = 0
                bracket = "]" if ch == "[" else "}"
                for j in range(i, len(text)):
                    if text[j] == ch:
                        depth += 1
                    elif text[j] == bracket:
                        depth -= 1
                    if depth == 0:
                        return text[i : j + 1]
        return text

    async def close(self):
        await self._client.aclose()
