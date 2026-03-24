import { useReducer, useEffect, useRef, useCallback } from 'react'
import { StoreContext, reducer, initialState } from './store'
import { listSessions, getMessages, connectWS } from './api'
import SessionList from './components/SessionList'
import Chat from './components/Chat'
import TreeView from './components/TreeView'
import Settings from './components/Settings'

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState)

  // Per-session WebSocket connections (persist across session switches)
  const wsMap = useRef<Map<string, ReturnType<typeof connectWS>>>(new Map())

  useEffect(() => {
    listSessions().then(s => dispatch({ type: 'SET_SESSIONS', sessions: s }))
  }, [])

  // Ensure active session has a WS connection
  const ensureWS = useCallback((sid: string) => {
    if (wsMap.current.has(sid)) return wsMap.current.get(sid)!

    const conn = connectWS(sid, {
      onChat(content) {
        dispatch({ type: 'SESSION_MSG', sid, message: { role: 'assistant', content, timestamp: Date.now() / 1000 } })
      },
      onThinking(sim, total) {
        dispatch({ type: 'SESSION_THINKING', sid, data: { sim, total } })
      },
      onTreeUpdate(tree) {
        dispatch({ type: 'SESSION_TREE', sid, tree })
      },
      onStrategy(strategy, tree) {
        dispatch({
          type: 'SESSION_MSG', sid,
          message: { role: 'assistant', content: strategy.analysis, timestamp: Date.now() / 1000, strategy, tree: tree || undefined },
        })
        if (tree) dispatch({ type: 'SESSION_TREE', sid, tree })
        dispatch({ type: 'SESSION_THINKING', sid, data: null })
      },
      onError(message) {
        dispatch({ type: 'SESSION_MSG', sid, message: { role: 'assistant', content: `Error: ${message}`, timestamp: Date.now() / 1000 } })
        dispatch({ type: 'SESSION_THINKING', sid, data: null })
      },
    })
    wsMap.current.set(sid, conn)
    return conn
  }, [])

  const sendMessage = useCallback((content: string, type: 'message' | 'analyze') => {
    if (!state.activeId) return
    const conn = ensureWS(state.activeId)
    dispatch({ type: 'SESSION_MSG', sid: state.activeId, message: { role: 'user', content, timestamp: Date.now() / 1000 } })
    if (type === 'analyze') dispatch({ type: 'SESSION_THINKING', sid: state.activeId, data: { sim: 0, total: state.config.simulations } })
    conn.send({ type, content, apiKey: state.config.apiKey, config: state.config })
  }, [state.activeId, state.config, ensureWS])

  const stopAnalysis = useCallback(() => {
    if (!state.activeId) return
    const conn = wsMap.current.get(state.activeId)
    if (conn) conn.send({ type: 'stop' })
    dispatch({ type: 'SESSION_THINKING', sid: state.activeId, data: null })
  }, [state.activeId])

  const selectSession = useCallback(async (id: string) => {
    const msgs = await getMessages(id)
    dispatch({ type: 'SET_ACTIVE', id, messages: msgs })
    // Pre-connect WS
    ensureWS(id)
  }, [ensureWS])

  // Derive current session's tree and thinking from sessionData
  const activeTree = state.activeId ? state.sessionData[state.activeId]?.tree ?? null : null
  const activeThinking = state.activeId ? state.sessionData[state.activeId]?.thinking ?? null : null

  return (
    <StoreContext.Provider value={{ state, dispatch }}>
      <div className="flex h-screen overflow-hidden">
        <SessionList onSelect={selectSession} />
        <Chat onSend={sendMessage} onStop={stopAnalysis} tree={activeTree} thinking={activeThinking} />
        <TreeView tree={activeTree} />
      </div>
      {state.settingsOpen && <Settings />}
    </StoreContext.Provider>
  )
}
