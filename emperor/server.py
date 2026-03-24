"""FastAPI server — REST + WebSocket for the Emperor web UI."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from .engine import Emperor, EmperorConfig
from .oracle import LLMOracle, LLMConfig
from .session import SessionStore, Message, Session
from .viz import json_tree

log = logging.getLogger("emperor")
app = FastAPI(title="Emperor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = SessionStore()

# ── REST: Sessions CRUD ───────────────────────────────────


@app.get("/api/sessions")
async def list_sessions():
    return store.list_all()


@app.post("/api/sessions")
async def create_session(body: dict | None = None):
    name = (body or {}).get("name", "")
    s = store.create(name)
    return {"id": s.id, "name": s.name}


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    return {"ok": store.delete(sid)}


@app.put("/api/sessions/{sid}")
async def update_session(sid: str, body: dict):
    if "name" in body:
        store.rename(sid, body["name"])
    return {"ok": True}


@app.get("/api/sessions/{sid}/messages")
async def get_messages(sid: str):
    s = store.get(sid)
    if not s:
        return JSONResponse({"error": "not found"}, 404)
    return [
        {
            "role": m.role,
            "content": m.content,
            "timestamp": m.timestamp,
            "strategy": m.strategy,
            "tree": m.tree_json,
        }
        for m in s.messages
    ]


# ── WebSocket: Real-time chat + analysis ──────────────────


@app.websocket("/ws/{sid}")
async def ws_endpoint(ws: WebSocket, sid: str):
    await ws.accept()
    session = store.get(sid)
    if not session:
        session = store.create()

    cancel_event = asyncio.Event()
    active_task: asyncio.Task | None = None

    async def receive_loop():
        """Receive messages; if 'stop' arrives during analysis, set cancel flag."""
        nonlocal active_task
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "message")

            if msg_type == "stop":
                cancel_event.set()
                if active_task and not active_task.done():
                    active_task.cancel()
                continue

            # Queue up for processing
            return data

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "message")

            if msg_type == "stop":
                cancel_event.set()
                if active_task and not active_task.done():
                    active_task.cancel()
                continue

            content = data.get("content", "").strip()
            api_key = data.get("apiKey", "")
            config = data.get("config", {})

            if not content:
                continue

            store.add_message(sid, Message(role="user", content=content))

            if msg_type == "analyze":
                cancel_event.clear()
                active_task = asyncio.create_task(
                    _handle_analyze(ws, session, content, api_key, config, cancel_event)
                )
                # Listen for stop while analysis runs
                while active_task and not active_task.done():
                    # Wait for either task completion or new message
                    recv_task = asyncio.create_task(ws.receive_json())
                    done, _ = await asyncio.wait(
                        [active_task, recv_task], return_when=asyncio.FIRST_COMPLETED
                    )
                    if recv_task in done:
                        msg = recv_task.result()
                        if msg.get("type") == "stop":
                            cancel_event.set()
                            active_task.cancel()
                            try:
                                await active_task
                            except (asyncio.CancelledError, Exception):
                                pass
                            store.add_message(sid, Message(role="assistant", content="Analysis stopped by user."))
                            await ws.send_json({"type": "chat", "content": "Analysis stopped."})
                            break
                        else:
                            recv_task.cancel()
                    else:
                        recv_task.cancel()
                active_task = None
            else:
                await _handle_chat(ws, session, content, api_key, config)

    except WebSocketDisconnect:
        if active_task and not active_task.done():
            active_task.cancel()
    except Exception as e:
        log.exception("WebSocket error")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


async def _handle_analyze(ws: WebSocket, session: Session, content: str, api_key: str, config: dict, cancel_event: asyncio.Event | None = None):
    """Run MCTS search with progress streaming."""
    cfg = _build_config(api_key, config)
    emperor = Emperor(cfg)

    async def on_progress(sim: int, total: int, root):
        # Check for cancellation
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError()
        await ws.send_json({"type": "thinking", "sim": sim, "total": total})
        if total > 0 and sim % max(total // 4, 1) == 0:
            await ws.send_json({"type": "tree_update", "tree": json_tree(root, max_depth=4)})

    try:
        # Parse players and goal from config
        players = config.get("players", ["You", "Opponent"])
        if isinstance(players, str):
            players = [p.strip() for p in players.split(",")]
        goal = config.get("goal", "")

        strategy = await emperor.analyze(
            content,
            players=players,
            goal=goal,
            on_progress=on_progress,
        )

        # Serialize strategy
        strat_data = {
            "bestAction": {"name": strategy.best_action.name, "description": strategy.best_action.description},
            "actionScores": [
                {"name": a.name, "description": a.description, "score": round(s, 3)}
                for a, s in strategy.action_scores
            ],
            "principalVariation": [a.name for a in strategy.principal_variation],
            "analysis": strategy.analysis,
            "confidence": round(strategy.confidence, 3),
            "vulnerabilities": strategy.vulnerabilities,
            "stats": strategy.stats,
        }

        tree_data = json_tree(emperor.last_root, max_depth=6) if hasattr(emperor, 'last_root') and emperor.last_root else None

        # Save assistant message
        store.add_message(session.id, Message(
            role="assistant",
            content=strategy.analysis,
            strategy=strat_data,
            tree_json=tree_data,
        ))

        await ws.send_json({"type": "strategy", "strategy": strat_data, "tree": tree_data})

    except Exception as e:
        log.exception("Analysis error")
        error_msg = f"Analysis failed: {e}"
        store.add_message(session.id, Message(role="assistant", content=error_msg))
        await ws.send_json({"type": "error", "message": error_msg})
    finally:
        await emperor.close()


async def _handle_chat(ws: WebSocket, session: Session, content: str, api_key: str, config: dict):
    """Direct conversation (no search)."""
    cfg = _build_config(api_key, config)
    oracle = LLMOracle(LLMConfig(
        provider=cfg.provider, model=cfg.model,
        api_key=cfg.api_key, temperature=cfg.temperature,
        reasoning=cfg.reasoning,
    ))

    try:
        # Build context from recent messages
        history = [
            {"role": m.role, "content": m.content}
            for m in session.messages[-10:]
        ]

        response = await oracle.chat(history)
        store.add_message(session.id, Message(role="assistant", content=response))
        await ws.send_json({"type": "chat", "content": response})

    except Exception as e:
        log.exception("Chat error")
        await ws.send_json({"type": "error", "message": str(e)})
    finally:
        await oracle.close()


def _build_config(api_key: str, config: dict) -> EmperorConfig:
    return EmperorConfig(
        provider=config.get("provider", "anthropic"),
        model=config.get("model", ""),
        api_key=api_key,
        temperature=config.get("temperature", 0.7),
        reasoning=config.get("reasoning", "standard"),
        simulations=config.get("simulations", 40),
        exploration=config.get("exploration", 1.41),
        max_depth=config.get("maxDepth", 8),
        max_actions=config.get("maxActions", 7),
        max_llm_calls=config.get("budget", 0),
        reflect=config.get("reflect", False),
        dirichlet_noise=config.get("dirichletNoise", True),
    )


# ── Static files (built frontend) ────────────────────────

_dist = Path(__file__).parent.parent / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True))


# ── Entry point ──────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
