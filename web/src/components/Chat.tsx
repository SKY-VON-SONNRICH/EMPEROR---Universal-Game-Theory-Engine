import { useRef, useEffect, useState, useMemo } from 'react'
import { useStore } from '../store'
import type { StrategyData, TreeNode } from '../types'

interface Props {
  onSend: (content: string, type: 'message' | 'analyze') => void
  onStop: () => void
  tree: TreeNode | null
  thinking: { sim: number; total: number } | null
}

// Map simulation progress to phase labels
function getPhase(sim: number, total: number): string {
  const pct = total > 0 ? sim / total : 0
  if (pct < 0.05) return 'Initializing search tree...'
  if (pct < 0.15) return 'Generating candidate strategies...'
  if (pct < 0.35) return 'Exploring action space...'
  if (pct < 0.55) return 'Evaluating positions...'
  if (pct < 0.75) return 'Deep-diving promising branches...'
  if (pct < 0.90) return 'Converging on best strategy...'
  if (pct < 1) return 'Finalizing analysis...'
  return 'Synthesizing report...'
}

export default function Chat({ onSend, onStop, thinking }: Props) {
  const { state } = useStore()
  const [input, setInput] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)
  const thinkingStartRef = useRef<number>(0)
  const [elapsed, setElapsed] = useState(0)

  // Auto-scroll
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [state.messages, thinking])

  // Elapsed time ticker while thinking
  useEffect(() => {
    if (thinking) {
      thinkingStartRef.current = Date.now()
      setElapsed(0)
      const iv = setInterval(() => setElapsed(Math.floor((Date.now() - thinkingStartRef.current) / 1000)), 1000)
      return () => clearInterval(iv)
    } else {
      setElapsed(0)
    }
  }, [!!thinking])

  const phase = useMemo(
    () => thinking ? getPhase(thinking.sim, thinking.total) : '',
    [thinking?.sim, thinking?.total]
  )

  const pct = thinking
    ? Math.round((thinking.sim / Math.max(thinking.total, 1)) * 100)
    : 0

  const handleSend = (type: 'message' | 'analyze') => {
    if (!input.trim()) return
    onSend(input.trim(), type)
    setInput('')
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend('message')
    }
  }

  // Empty state
  if (!state.activeId) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-600">
        <div className="text-center">
          <div className="text-5xl mb-4">♛</div>
          <div className="text-xl font-light tracking-wide">Emperor</div>
          <div className="text-sm mt-2 text-gray-700">Select or create a session to begin</div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col min-w-0">
      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4">
        {state.messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] rounded-lg px-4 py-2.5 text-sm leading-relaxed ${
              m.role === 'user'
                ? 'bg-blue-600/20 text-blue-100 border border-blue-800/30'
                : 'bg-gray-800/60 text-gray-200 border border-gray-700/30'
            }`}>
              <div className="whitespace-pre-wrap">{m.content}</div>
              {m.strategy && <StrategyCard strategy={m.strategy} />}
            </div>
          </div>
        ))}

        {/* ── Thinking Overlay ── */}
        {thinking && (
          <div className="flex justify-start">
            <div className="bg-gray-800/60 backdrop-blur border border-gray-700/40 rounded-xl px-6 py-5 w-80">
              {/* Spinner + Phase */}
              <div className="flex items-center gap-4 mb-4">
                <div className="relative flex-shrink-0">
                  <div className="spinner-orbit" />
                  <div className="pulse-ring" />
                </div>
                <div>
                  <div className="text-sm font-medium text-gray-200 phase-text">{phase}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{elapsed}s elapsed</div>
                </div>
              </div>

              {/* Progress bar */}
              <div className="mb-3">
                <div className="flex justify-between text-xs text-gray-500 mb-1">
                  <span>Simulation {thinking.sim} / {thinking.total}</span>
                  <span>{pct}%</span>
                </div>
                <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full transition-all duration-500 ease-out"
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>

              {/* Stop button */}
              <button
                onClick={onStop}
                className="w-full py-2 text-xs font-medium text-red-400 hover:text-red-300 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 hover:border-red-500/30 rounded-lg transition"
              >
                ■ Stop Analysis
              </button>
            </div>
          </div>
        )}
      </div>

      {/* ── Input ── */}
      <div className="p-3 border-t border-gray-800 bg-gray-950">
        {!state.config.apiKey && (
          <div className="mb-2 text-xs text-amber-400/80 px-1">
            Set your API key in Settings to start analyzing.
          </div>
        )}
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe a strategic situation..."
            rows={2}
            disabled={!!thinking}
            className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:border-blue-600 placeholder:text-gray-600 disabled:opacity-40"
          />
          <div className="flex flex-col gap-1.5">
            {thinking ? (
              <button
                onClick={onStop}
                className="px-3 py-3.5 text-xs font-medium text-red-400 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 rounded transition"
              >
                ■ Stop
              </button>
            ) : (
              <>
                <button
                  onClick={() => handleSend('message')}
                  disabled={!input.trim() || !state.config.apiKey}
                  className="px-3 py-1.5 text-xs bg-gray-700 hover:bg-gray-600 disabled:opacity-30 rounded transition"
                >
                  Send
                </button>
                <button
                  onClick={() => handleSend('analyze')}
                  disabled={!input.trim() || !state.config.apiKey}
                  className="px-3 py-1.5 text-xs bg-blue-600 hover:bg-blue-500 disabled:opacity-30 rounded transition font-medium"
                >
                  Analyze
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function StrategyCard({ strategy }: { strategy: StrategyData }) {
  return (
    <div className="mt-3 p-3 bg-gray-900/80 rounded border border-gray-700/40 text-xs space-y-2">
      <div className="flex items-center justify-between">
        <span className="font-medium text-blue-300">Best: {strategy.bestAction.name}</span>
        <span className="text-gray-400">Confidence: {(strategy.confidence * 100).toFixed(0)}%</span>
      </div>
      <div className="h-1 bg-gray-700 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all"
          style={{
            width: `${strategy.confidence * 100}%`,
            background: strategy.confidence > 0.7 ? '#22c55e' : strategy.confidence > 0.4 ? '#eab308' : '#ef4444',
          }}
        />
      </div>
      <div className="space-y-1">
        {strategy.actionScores.slice(0, 4).map((a, i) => (
          <div key={i} className="flex items-center gap-2">
            <div className="w-24 truncate text-gray-300">{a.name}</div>
            <div className="flex-1 h-1 bg-gray-700 rounded-full overflow-hidden">
              <div className="h-full bg-blue-500/60 rounded-full" style={{ width: `${a.score * 100}%` }} />
            </div>
            <div className="w-8 text-right text-gray-500">{(a.score * 100).toFixed(0)}%</div>
          </div>
        ))}
      </div>
      {strategy.vulnerabilities.length > 0 && (
        <div className="pt-1 border-t border-gray-700/40">
          <div className="text-amber-400/80 font-medium mb-1">Vulnerabilities:</div>
          {strategy.vulnerabilities.map((v, i) => (
            <div key={i} className="text-gray-400 pl-2">• {v}</div>
          ))}
        </div>
      )}
      <div className="flex gap-3 text-gray-500 pt-1">
        {strategy.stats.simulations != null && <span>{strategy.stats.simulations} sims</span>}
        {strategy.stats.llm_calls != null && <span>{strategy.stats.llm_calls} LLM calls</span>}
        {strategy.stats.elapsed_sec != null && <span>{strategy.stats.elapsed_sec}s</span>}
      </div>
    </div>
  )
}
