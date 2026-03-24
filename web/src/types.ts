export interface Session {
  id: string
  name: string
  messageCount: number
  createdAt: number
  updatedAt: number
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  strategy?: StrategyData
  tree?: TreeNode
}

export interface StrategyData {
  bestAction: { name: string; description: string }
  actionScores: { name: string; description: string; score: number }[]
  principalVariation: string[]
  analysis: string
  confidence: number
  vulnerabilities: string[]
  stats: Record<string, number>
}

export interface TreeNode {
  action: string
  description?: string
  visits: number
  q: Record<string, number>
  prior?: number
  children?: TreeNode[]
}

export interface Config {
  provider: string
  model: string
  reasoning: string   // "standard" | "extended"
  apiKey: string
  backendUrl: string  // e.g. "https://your-server.com" — leave empty for same-origin
  simulations: number
  maxDepth: number
  maxActions: number
  exploration: number
  reflect: boolean
  budget: number
  temperature: number
}

export const DEFAULT_CONFIG: Config = {
  provider: 'anthropic',
  model: 'claude-opus-4-20250514',
  reasoning: 'standard',
  apiKey: '',
  backendUrl: '',
  simulations: 40,
  maxDepth: 8,
  maxActions: 7,
  exploration: 1.41,
  reflect: false,
  budget: 0,
  temperature: 0.7,
}

// Model catalog per provider
export const MODEL_OPTIONS: Record<string, { value: string; label: string; modes: string[] }[]> = {
  anthropic: [
    { value: 'claude-opus-4-20250514', label: 'Claude Opus 4.6', modes: ['standard', 'extended'] },
  ],
  openai: [
    { value: 'gpt-5.1', label: 'GPT-5.1', modes: ['thinking', 'pro'] },
    { value: 'gpt-5.2', label: 'GPT-5.2', modes: ['thinking', 'pro'] },
    { value: 'gpt-5.4', label: 'GPT-5.4', modes: ['thinking', 'pro'] },
  ],
}

// Reasoning modes differ by provider
export const REASONING_MODES: Record<string, { value: string; label: string; description: string }[]> = {
  anthropic: [
    { value: 'standard', label: 'Standard', description: 'Normal inference, fast and efficient' },
    { value: 'extended', label: 'Extended Thinking', description: 'Deep chain-of-thought reasoning (slower, higher quality)' },
  ],
  openai: [
    { value: 'thinking', label: 'Thinking', description: 'Standard reasoning mode' },
    { value: 'pro', label: 'Pro', description: 'Maximum depth reasoning (slower, highest quality)' },
  ],
}
