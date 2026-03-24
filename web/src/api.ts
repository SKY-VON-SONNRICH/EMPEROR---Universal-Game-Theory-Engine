import type { Session, Message, TreeNode, StrategyData } from './types'

/** Get saved backend URL from localStorage (mirrors store.ts logic) */
function getBackendUrl(): string {
  try {
    const saved = localStorage.getItem('emperor_config')
    if (saved) {
      const config = JSON.parse(saved)
      return (config.backendUrl || '').replace(/\/+$/, '')
    }
  } catch { /* ignore */ }
  return ''
}

export async function listSessions(): Promise<Session[]> {
  const base = getBackendUrl()
  const r = await fetch(`${base}/api/sessions`)
  return r.json()
}

export async function createSession(name?: string): Promise<{ id: string; name: string }> {
  const base = getBackendUrl()
  const r = await fetch(`${base}/api/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
  return r.json()
}

export async function deleteSession(id: string): Promise<void> {
  const base = getBackendUrl()
  await fetch(`${base}/api/sessions/${id}`, { method: 'DELETE' })
}

export async function renameSession(id: string, name: string): Promise<void> {
  const base = getBackendUrl()
  await fetch(`${base}/api/sessions/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
}

export async function getMessages(id: string): Promise<Message[]> {
  const base = getBackendUrl()
  const r = await fetch(`${base}/api/sessions/${id}/messages`)
  return r.json()
}

/** Check if the backend is reachable */
export async function checkBackend(): Promise<boolean> {
  try {
    const base = getBackendUrl()
    const r = await fetch(`${base}/api/sessions`, { signal: AbortSignal.timeout(5000) })
    return r.ok
  } catch {
    return false
  }
}

export interface WSHandlers {
  onChat: (content: string) => void
  onThinking: (sim: number, total: number) => void
  onTreeUpdate: (tree: TreeNode) => void
  onStrategy: (strategy: StrategyData, tree: TreeNode | null) => void
  onError: (message: string) => void
}

export function connectWS(sessionId: string, handlers: WSHandlers) {
  const backendUrl = getBackendUrl()
  let wsUrl: string

  if (backendUrl) {
    // Convert http(s) URL to ws(s) URL
    const url = new URL(backendUrl)
    const proto = url.protocol === 'https:' ? 'wss:' : 'ws:'
    wsUrl = `${proto}//${url.host}/ws/${sessionId}`
  } else {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
    wsUrl = `${proto}//${location.host}/ws/${sessionId}`
  }

  const ws = new WebSocket(wsUrl)

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
