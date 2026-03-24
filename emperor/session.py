"""Session management — multi-session chat with tree state."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .core import Node


@dataclass
class Message:
    role: str              # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    strategy: dict | None = None   # serialized Strategy
    tree_json: dict | None = None  # json_tree snapshot


@dataclass
class Session:
    id: str
    name: str
    messages: list[Message] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    # In-memory only (not persisted)
    tree_root: Node | None = field(default=None, repr=False)


class SessionStore:
    """In-memory session store with JSON file persistence."""

    def __init__(self, path: str | Path = "~/.emperor/sessions.json"):
        self._sessions: dict[str, Session] = {}
        self._path = Path(path).expanduser()
        self._load()

    def create(self, name: str = "") -> Session:
        sid = uuid.uuid4().hex[:12]
        name = name or f"Session {len(self._sessions) + 1}"
        session = Session(id=sid, name=name)
        self._sessions[sid] = session
        self._save()
        return session

    def get(self, sid: str) -> Session | None:
        return self._sessions.get(sid)

    def list_all(self) -> list[dict]:
        return sorted(
            [
                {
                    "id": s.id,
                    "name": s.name,
                    "messageCount": len(s.messages),
                    "createdAt": s.created_at,
                    "updatedAt": s.messages[-1].timestamp if s.messages else s.created_at,
                }
                for s in self._sessions.values()
            ],
            key=lambda x: x["updatedAt"],
            reverse=True,
        )

    def delete(self, sid: str) -> bool:
        if sid in self._sessions:
            del self._sessions[sid]
            self._save()
            return True
        return False

    def rename(self, sid: str, name: str) -> bool:
        s = self._sessions.get(sid)
        if s:
            s.name = name
            self._save()
            return True
        return False

    def add_message(self, sid: str, msg: Message) -> None:
        s = self._sessions.get(sid)
        if s:
            s.messages.append(msg)
            self._save()

    # ── Persistence ────────────────────────────────────────

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for sid, s in self._sessions.items():
            data[sid] = {
                "id": s.id,
                "name": s.name,
                "config": s.config,
                "created_at": s.created_at,
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "timestamp": m.timestamp,
                        "strategy": m.strategy,
                        "tree_json": m.tree_json,
                    }
                    for m in s.messages
                ],
            }
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def _load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for sid, d in data.items():
                msgs = [
                    Message(
                        role=m["role"],
                        content=m["content"],
                        timestamp=m.get("timestamp", 0),
                        strategy=m.get("strategy"),
                        tree_json=m.get("tree_json"),
                    )
                    for m in d.get("messages", [])
                ]
                self._sessions[sid] = Session(
                    id=d["id"],
                    name=d["name"],
                    messages=msgs,
                    config=d.get("config", {}),
                    created_at=d.get("created_at", 0),
                )
        except (json.JSONDecodeError, KeyError):
            pass
