import type { Session, Message, TreeNode, StrategyData } from './types'

const BASE = ''

export async function listSessions(): Promise<Session[]> {
  const r = await fetch(`${BASE}/api/sessions`)
  return r.json()
}

export async function createSession(name?: string): Promise<{ id: string; name: string }> {
  const r = await fetch(`${BASE}/api/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
  return r.json()
}

export async function deleteSession(id: string): Promise<void> {
  await fetch(`${BASE}/api/sessions/${id}`, { method: 'DELETE' })
}

export async function renameSession(id: string, name: string): Promise<void> {
  await fetch(`${BASE}/api/sessions/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
}

export async function getMessages(id: string): Promise<Message[]> {
  const r = await fetch(`${BASE}/api/sessions/${id}/messages`)
  return r.json()
}

export interface WSHandlers {
  onChat: (content: string) => void
  onThinking: (sim: number, total: number) => void
  onTreeUpdate: (tree: TreeNode) => void
  onStrategy: (strategy: StrategyData, tree: TreeNode | null) => void
  onError: (message: string) => void
}

export function connectWS(sessionId: string, handlers: WSHandlers) {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
  const ws = new WebSocket(`${proto}//${location.host}/ws/${sessionId}`)

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data)
    switch (msg.type) {
      case 'chat': handlers.onChat(msg.content); break
      case 'thinking': handlers.onThinking(msg.sim, msg.total); break
      case 'tree_update': handlers.onTreeUpdate(msg.tree); break
      case 'strategy': handlers.onStrategy(msg.strategy, msg.tree); break
      case 'error': handlers.onError(msg.message); break
    }
  }

  return {
    send: (msg: Record<string, unknown>) => {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg))
    },
    close: () => ws.close(),
    ws,
  }
}
