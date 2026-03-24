import { useStore } from '../store'
import { createSession, deleteSession, listSessions } from '../api'

export default function SessionList({ onSelect }: { onSelect: (id: string) => void }) {
  const { state, dispatch } = useStore()

  const handleCreate = async () => {
    const s = await createSession()
    const sessions = await listSessions()
    dispatch({ type: 'SET_SESSIONS', sessions })
    onSelect(s.id)
  }

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    await deleteSession(id)
    const sessions = await listSessions()
    dispatch({ type: 'SET_SESSIONS', sessions })
    if (state.activeId === id) dispatch({ type: 'SET_ACTIVE', id: null })
  }

  return (
    <div className="w-64 border-r border-gray-800 flex flex-col bg-gray-950">
      {/* Header */}
      <div className="p-3 border-b border-gray-800 flex items-center justify-between">
        <span className="font-semibold text-sm tracking-wide text-gray-300">EMPEROR</span>
        <button
          onClick={handleCreate}
          className="px-2 py-1 text-xs bg-gray-800 hover:bg-gray-700 rounded transition"
        >
          + New
        </button>
      </div>

      {/* Sessions */}
      <div className="flex-1 overflow-y-auto">
        {state.sessions.map(s => (
          <div
            key={s.id}
            onClick={() => onSelect(s.id)}
            className={`group px-3 py-2.5 cursor-pointer border-b border-gray-900 flex items-center justify-between transition ${
              state.activeId === s.id ? 'bg-gray-800/60' : 'hover:bg-gray-900'
            }`}
          >
            <div className="min-w-0 flex-1">
              <div className="text-sm truncate">{s.name}</div>
              <div className="text-xs text-gray-500">{s.messageCount} messages</div>
            </div>
            <button
              onClick={(e) => handleDelete(e, s.id)}
              className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-400 text-xs px-1 transition"
            >
              ✕
            </button>
          </div>
        ))}
        {state.sessions.length === 0 && (
          <div className="p-4 text-center text-gray-600 text-sm">
            No sessions yet.<br />Click "+ New" to start.
          </div>
        )}
      </div>

      {/* Settings */}
      <div className="p-3 border-t border-gray-800">
        <button
          onClick={() => dispatch({ type: 'TOGGLE_SETTINGS' })}
          className="w-full text-left text-sm text-gray-400 hover:text-gray-200 transition flex items-center gap-2"
        >
          <span>⚙</span> Settings
        </button>
      </div>
    </div>
  )
}
