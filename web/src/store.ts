import { createContext, useContext, type Dispatch } from 'react'
import type { Session, Message, TreeNode, Config } from './types'
import { DEFAULT_CONFIG } from './types'

// Per-session ephemeral data (tree, thinking state)
export interface SessionData {
  tree: TreeNode | null
  thinking: { sim: number; total: number } | null
}

export interface AppState {
  sessions: Session[]
  activeId: string | null
  messages: Message[]
  sessionData: Record<string, SessionData>  // keyed by session id
  settingsOpen: boolean
  config: Config
}

export type Action =
  | { type: 'SET_SESSIONS'; sessions: Session[] }
  | { type: 'SET_ACTIVE'; id: string | null; messages?: Message[] }
  | { type: 'ADD_MESSAGE'; message: Message }
  | { type: 'SESSION_MSG'; sid: string; message: Message }
  | { type: 'SESSION_TREE'; sid: string; tree: TreeNode | null }
  | { type: 'SESSION_THINKING'; sid: string; data: { sim: number; total: number } | null }
  | { type: 'TOGGLE_SETTINGS' }
  | { type: 'UPDATE_CONFIG'; config: Partial<Config> }

function loadConfig(): Config {
  try {
    const saved = localStorage.getItem('emperor_config')
    return saved ? { ...DEFAULT_CONFIG, ...JSON.parse(saved) } : DEFAULT_CONFIG
  } catch {
    return DEFAULT_CONFIG
  }
}

function getSD(state: AppState, sid: string): SessionData {
  return state.sessionData[sid] || { tree: null, thinking: null }
}

export const initialState: AppState = {
  sessions: [],
  activeId: null,
  messages: [],
  sessionData: {},
  settingsOpen: false,
  config: loadConfig(),
}

export function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_SESSIONS':
      return { ...state, sessions: action.sessions }

    case 'SET_ACTIVE': {
      // Restore tree from last strategy message
      const msgs = action.messages || []
      const lastTree = [...msgs].reverse().find(m => m.tree)?.tree ?? null
      const sd = getSD(state, action.id || '')
      return {
        ...state,
        activeId: action.id,
        messages: msgs,
        sessionData: action.id ? {
          ...state.sessionData,
          [action.id]: { ...sd, tree: sd.tree || lastTree },
        } : state.sessionData,
      }
    }

    case 'ADD_MESSAGE':
      return { ...state, messages: [...state.messages, action.message] }

    case 'SESSION_MSG': {
      const isActive = action.sid === state.activeId
      const newState = isActive
        ? { ...state, messages: [...state.messages, action.message] }
        : state
      // If message has tree, store in sessionData
      if (action.message.tree) {
        return {
          ...newState,
          sessionData: {
            ...newState.sessionData,
            [action.sid]: { ...getSD(newState, action.sid), tree: action.message.tree },
          },
        }
      }
      return newState
    }

    case 'SESSION_TREE':
      return {
        ...state,
        sessionData: {
          ...state.sessionData,
          [action.sid]: { ...getSD(state, action.sid), tree: action.tree },
        },
      }

    case 'SESSION_THINKING':
      return {
        ...state,
        sessionData: {
          ...state.sessionData,
          [action.sid]: { ...getSD(state, action.sid), thinking: action.data },
        },
      }

    case 'TOGGLE_SETTINGS':
      return { ...state, settingsOpen: !state.settingsOpen }

    case 'UPDATE_CONFIG': {
      const config = { ...state.config, ...action.config }
      localStorage.setItem('emperor_config', JSON.stringify(config))
      return { ...state, config }
    }

    default:
      return state
  }
}

export const StoreContext = createContext<{ state: AppState; dispatch: Dispatch<Action> }>({
  state: initialState,
  dispatch: () => {},
})

export const useStore = () => useContext(StoreContext)
