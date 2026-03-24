import { useState } from 'react'
import { useStore } from '../store'
import { DEFAULT_CONFIG, MODEL_OPTIONS, REASONING_MODES } from '../types'
import type { Config } from '../types'

export default function Settings() {
  const { state, dispatch } = useStore()
  const [draft, setDraft] = useState<Config>({ ...state.config })
  const [showKey, setShowKey] = useState(false)

  const update = (patch: Partial<Config>) => setDraft(d => ({ ...d, ...patch }))

  const models = MODEL_OPTIONS[draft.provider] || []
  const selectedModel = models.find(m => m.value === draft.model)
  const reasoningModes = REASONING_MODES[draft.provider] || []
  const availableModes = selectedModel?.modes || reasoningModes.map(r => r.value)

  const save = () => {
    dispatch({ type: 'UPDATE_CONFIG', config: draft })
    dispatch({ type: 'TOGGLE_SETTINGS' })
  }

  const reset = () => setDraft({ ...DEFAULT_CONFIG, apiKey: draft.apiKey })

  const onProviderChange = (provider: string) => {
    const defaultModel = MODEL_OPTIONS[provider]?.[0]?.value || ''
    const defaultMode = REASONING_MODES[provider]?.[0]?.value || 'standard'
    update({ provider, model: defaultModel, reasoning: defaultMode })
  }

  const onModelChange = (model: string) => {
    const m = models.find(x => x.value === model)
    const reasoning = m?.modes?.includes(draft.reasoning) ? draft.reasoning : (m?.modes?.[0] || 'standard')
    update({ model, reasoning })
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={() => dispatch({ type: 'TOGGLE_SETTINGS' })}>
      <div className="bg-gray-900 border border-gray-700 rounded-xl w-[460px] max-h-[85vh] overflow-y-auto shadow-xl" onClick={e => e.stopPropagation()}>
        <div className="p-4 border-b border-gray-800 flex items-center justify-between">
          <h2 className="font-semibold">Settings</h2>
          <button onClick={() => dispatch({ type: 'TOGGLE_SETTINGS' })} className="text-gray-500 hover:text-gray-300">✕</button>
        </div>

        <div className="p-4 space-y-4">
          {/* API Key */}
          <Field label="API Key">
            <div className="flex gap-1">
              <input
                type={showKey ? 'text' : 'password'}
                value={draft.apiKey}
                onChange={e => update({ apiKey: e.target.value })}
                placeholder="sk-ant-... or sk-..."
                className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm focus:outline-none focus:border-blue-600"
              />
              <button onClick={() => setShowKey(!showKey)} className="text-xs text-gray-500 px-2">{showKey ? 'Hide' : 'Show'}</button>
            </div>
            <div className="text-xs text-gray-600 mt-1">Stored in browser only. Never sent to our servers.</div>
          </Field>

          {/* Backend URL */}
          <Field label="Backend URL">
            <input
              value={draft.backendUrl}
              onChange={e => update({ backendUrl: e.target.value })}
              placeholder="https://your-server.com (leave empty for same-origin)"
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm focus:outline-none focus:border-blue-600"
            />
            <div className="text-xs text-gray-600 mt-1">
              URL of your Emperor backend. Required when using GitHub Pages.
            </div>
          </Field>

          {/* Provider */}
          <Field label="Provider">
            <div className="flex gap-2">
              {['anthropic', 'openai'].map(p => (
                <button
                  key={p}
                  onClick={() => onProviderChange(p)}
                  className={`flex-1 py-2 rounded text-sm font-medium transition ${
                    draft.provider === p
                      ? 'bg-blue-600/20 border border-blue-500/40 text-blue-300'
                      : 'bg-gray-800 border border-gray-700 text-gray-400 hover:border-gray-600'
                  }`}
                >
                  {p === 'anthropic' ? 'Anthropic' : 'OpenAI'}
                </button>
              ))}
            </div>
          </Field>

          {/* Model */}
          <Field label="Model">
            <div className="grid gap-1.5">
              {models.map(m => (
                <button
                  key={m.value}
                  onClick={() => onModelChange(m.value)}
                  className={`flex items-center justify-between px-3 py-2 rounded text-sm transition ${
                    draft.model === m.value
                      ? 'bg-blue-600/20 border border-blue-500/40 text-blue-200'
                      : 'bg-gray-800 border border-gray-700 text-gray-300 hover:border-gray-600'
                  }`}
                >
                  <span>{m.label}</span>
                  <span className="text-xs text-gray-500 font-mono">{m.value}</span>
                </button>
              ))}
            </div>
            <input
              value={draft.model}
              onChange={e => update({ model: e.target.value })}
              placeholder="Or type custom model ID..."
              className="mt-2 w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs text-gray-400 focus:outline-none focus:border-blue-600"
            />
          </Field>

          {/* Reasoning Mode */}
          <Field label="Reasoning Mode">
            <div className="flex gap-2">
              {reasoningModes.map(r => (
                <button
                  key={r.value}
                  onClick={() => update({ reasoning: r.value })}
                  disabled={!availableModes.includes(r.value)}
                  className={`flex-1 py-2 px-2 rounded text-sm transition ${
                    draft.reasoning === r.value
                      ? 'bg-purple-600/20 border border-purple-500/40 text-purple-300'
                      : availableModes.includes(r.value)
                        ? 'bg-gray-800 border border-gray-700 text-gray-400 hover:border-gray-600'
                        : 'bg-gray-800/50 border border-gray-800 text-gray-600 cursor-not-allowed'
                  }`}
                >
                  {r.label}
                </button>
              ))}
            </div>
            <div className="text-xs text-gray-600 mt-1">
              {reasoningModes.find(r => r.value === draft.reasoning)?.description}
            </div>
            {(draft.reasoning === 'extended' || draft.reasoning === 'pro') && (
              <div className="text-xs text-amber-400/70 mt-1">
                Deep reasoning uses more tokens and takes longer, but produces significantly better strategic analysis.
              </div>
            )}
          </Field>

          <div className="grid grid-cols-2 gap-3">
            <Field label="Simulations">
              <NumberInput value={draft.simulations} min={5} max={500} onChange={v => update({ simulations: v })} />
            </Field>
            <Field label="Max Depth">
              <NumberInput value={draft.maxDepth} min={2} max={20} onChange={v => update({ maxDepth: v })} />
            </Field>
            <Field label="Max Actions">
              <NumberInput value={draft.maxActions} min={2} max={15} onChange={v => update({ maxActions: v })} />
            </Field>
            <Field label="Budget (0=∞)">
              <NumberInput value={draft.budget} min={0} max={1000} onChange={v => update({ budget: v })} />
            </Field>
            <Field label="Temperature">
              <input
                type="range"
                min={0} max={1} step={0.1}
                value={draft.temperature}
                onChange={e => update({ temperature: parseFloat(e.target.value) })}
                className="w-full accent-blue-500"
              />
              <div className="text-xs text-gray-500 text-center">{draft.temperature}</div>
            </Field>
            <Field label="Exploration (C)">
              <input
                type="range"
                min={0.5} max={3} step={0.1}
                value={draft.exploration}
                onChange={e => update({ exploration: parseFloat(e.target.value) })}
                className="w-full accent-blue-500"
              />
              <div className="text-xs text-gray-500 text-center">{draft.exploration.toFixed(1)}</div>
            </Field>
          </div>

          {/* Reflection toggle */}
          <div className="flex items-center justify-between py-1">
            <div>
              <div className="text-sm">Reflection</div>
              <div className="text-xs text-gray-500">LLM critiques strategy after search</div>
            </div>
            <button
              onClick={() => update({ reflect: !draft.reflect })}
              className={`w-10 h-5 rounded-full transition-colors ${draft.reflect ? 'bg-blue-600' : 'bg-gray-700'}`}
            >
              <div className={`w-4 h-4 rounded-full bg-white transition-transform ${draft.reflect ? 'translate-x-5' : 'translate-x-0.5'}`} />
            </button>
          </div>
        </div>

        {/* Actions */}
        <div className="p-4 border-t border-gray-800 flex justify-between">
          <button onClick={reset} className="text-sm text-gray-500 hover:text-gray-300">Reset defaults</button>
          <div className="flex gap-2">
            <button onClick={() => dispatch({ type: 'TOGGLE_SETTINGS' })} className="px-3 py-1.5 text-sm bg-gray-800 hover:bg-gray-700 rounded">Cancel</button>
            <button onClick={save} className="px-3 py-1.5 text-sm bg-blue-600 hover:bg-blue-500 rounded font-medium">Save</button>
          </div>
        </div>
      </div>
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-sm text-gray-400 mb-1">{label}</label>
      {children}
    </div>
  )
}

function NumberInput({ value, min, max, onChange }: { value: number; min: number; max: number; onChange: (v: number) => void }) {
  return (
    <input
      type="number"
      value={value}
      min={min}
      max={max}
      onChange={e => onChange(parseInt(e.target.value) || min)}
      className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm focus:outline-none focus:border-blue-600"
    />
  )
}
