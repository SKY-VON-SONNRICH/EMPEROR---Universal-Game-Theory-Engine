import { useReducer, useEffect, useRef, useCallback, useState } from 'react'
import { StoreContext, reducer, initialState } from './store'
import { listSessions, getMessages, connectWS, checkBackend } from './api'
import SessionList from './components/SessionList'
import Chat from './components/Chat'
import TreeView from './components/TreeView'
import Settings from './components/Settings'

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState)
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null) // null = checking

  // Per-session WebSocket connections (persist across session switches)
  const wsMap = useRef<Map<string, ReturnType<typeof connectWS>>>(new Map())

  // Check backend connectivity
  useEffect(() => {
    setBackendOnline(null)
    checkBackend().then(ok => {
      setBackendOnline(ok)
      if (ok) listSessions().then(s => dispatch({ type: 'SET_SESSIONS', sessions: s }))
    })
  }, [state.config.backendUrl])

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

  // Show landing page when backend is offline
  if (backendOnline === false) {
    return (
      <StoreContext.Provider value={{ state, dispatch }}>
        <div className="flex h-screen items-center justify-center bg-gray-950">
          <div className="max-w-lg text-center px-6">
            <div className="text-6xl mb-6">♛</div>
            <h1 className="text-3xl font-light tracking-wide mb-2">Emperor</h1>
            <p className="text-gray-500 mb-8">Universal Game Theory Engine</p>

            <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 text-left space-y-4">
              <div>
                <h2 className="text-sm font-medium text-gray-300 mb-2">Getting Started</h2>
                <p className="text-sm text-gray-500 leading-relaxed">
                  Emperor needs a backend server to run game-theoretic analysis.
                  {state.config.backendUrl
                    ? ' Could not connect to: ' + state.config.backendUrl
                    : ' No backend URL configured.'}
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="text-xs font-medium text-gray-400">Option 1: Connect to a hosted backend</h3>
                <p className="text-xs text-gray-600">
                  Click Settings below and enter your backend URL.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="text-xs font-medium text-gray-400">Option 2: Run your own backend</h3>
                <div className="bg-gray-950 border border-gray-800 rounded-lg p-3 font-mono text-xs text-gray-400 space-y-1">
                  <div><span className="text-blue-400">$</span> pip install fastapi uvicorn websockets httpx</div>
                  <div><span className="text-blue-400">$</span> python -m emperor</div>
                </div>
                <p className="text-xs text-gray-600">
                  Then set backend URL to <span className="font-mono text-gray-500">http://localhost:8000</span>
                </p>
              </div>

              <div className="flex gap-2 pt-2">
                <button
                  onClick={() => dispatch({ type: 'TOGGLE_SETTINGS' })}
                  className="flex-1 py-2 text-sm bg-blue-600 hover:bg-blue-500 rounded-lg font-medium transition"
                >
                  Open Settings
                </button>
                <button
                  onClick={() => checkBackend().then(ok => {
                    setBackendOnline(ok)
                    if (ok) listSessions().then(s => dispatch({ type: 'SET_SESSIONS', sessions: s }))
                  })}
                  className="px-4 py-2 text-sm bg-gray-800 hover:bg-gray-700 rounded-lg transition"
                >
                  Retry
                </button>
              </div>
            </div>
          </div>
        </div>
        {state.settingsOpen && <Settings />}
      </StoreContext.Provider>
    )
  }

  // Show loading state
  if (backendOnline === null) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-950">
        <div className="text-center">
          <div className="spinner-orbit mx-auto mb-4" />
          <div className="text-gray-500 text-sm">Connecting to backend...</div>
        </div>
      </div>
    )
  }

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
