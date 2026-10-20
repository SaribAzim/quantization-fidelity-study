import { useCallback, useEffect, useRef, useState } from 'react'
import { AnswerCard, useReplay } from '../components/Replay'
import { BEST_TWO, FAMILY_COLOR, modelById, useData } from '../data'
import { ModelMeta, compareFoot } from './Showcase'

const API = 'http://127.0.0.1:8000'
const SPEEDS = [{ label: 'Real speed', speed: 1 }, { label: '4 times', speed: 4 }, { label: '16 times', speed: 16 }]
const TITLES: Record<string, string> = { g128_l0131: 'Closest to the original', per_row_vkod: 'Best exam score' }

type LiveState = { status: 'idle' | 'waiting' | 'queued' | 'loading' | 'streaming' | 'done' | 'error'; text: string; tokens: number; tps: number | null; message?: string }
const idle = (): LiveState => ({ status: 'idle', text: '', tokens: 0, tps: null })

/** Asks the local backend (deployment/server.py) whether it is running. Never called automatically on the public site. */
async function probe(): Promise<boolean> {
  try {
    const ctrl = new AbortController()
    const t = window.setTimeout(() => ctrl.abort(), 1500)
    const r = await fetch(`${API}/api/health`, { signal: ctrl.signal })
    window.clearTimeout(t)
    return r.ok
  } catch {
    return false
  }
}

async function streamAnswer(model: string, prompt: string, signal: AbortSignal, onEvent: (e: Record<string, unknown>) => void) {
  const r = await fetch(`${API}/api/generate`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, signal,
    body: JSON.stringify({ model, prompt, maxNewTokens: 120 }),
  })
  if (!r.ok || !r.body) throw new Error(r.status === 422 ? 'The question is empty or longer than 2,000 characters.' : `The backend answered with status ${r.status}.`)
  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let cut
    while ((cut = buffer.indexOf('\n\n')) >= 0) {
      const chunk = buffer.slice(0, cut)
      buffer = buffer.slice(cut + 2)
      if (chunk.startsWith('data: ')) onEvent(JSON.parse(chunk.slice(6)))
    }
  }
}

function statusLine(s: LiveState): string {
  switch (s.status) {
    case 'waiting': return 'Waiting for the other model to finish.'
    case 'queued': return 'Waiting for the graphics card.'
    case 'loading': return 'Loading this model onto the graphics card, about 20 seconds.'
    case 'streaming': return 'Answering.'
    case 'done': return `Finished: ${s.tokens} word pieces at ${s.tps ?? 0} a second.`
    case 'error': return s.message ?? 'Something went wrong.'
    default: return 'Ready.'
  }
}

function LivePanel() {
  const { results } = useData()
  const [question, setQuestion] = useState('')
  const [states, setStates] = useState<LiveState[]>([idle(), idle()])
  const [running, setRunning] = useState(false)
  const abort = useRef<AbortController | null>(null)

  const update = (i: number, patch: Partial<LiveState>) => setStates((prev) => prev.map((s, k) => (k === i ? { ...s, ...patch } : s)))

  const ask = async () => {
    const prompt = question.trim()
    if (!prompt) return
    const ctrl = new AbortController()
    abort.current = ctrl
    setRunning(true)
    setStates([{ ...idle(), status: 'waiting' }, { ...idle(), status: 'waiting' }])
    try {
      for (let i = 0; i < BEST_TWO.length; i++) {
        update(i, { status: 'queued' })
        await streamAnswer(BEST_TWO[i], prompt, ctrl.signal, (e) => {
          if (e.type === 'loading') update(i, { status: 'loading' })
          if (e.type === 'ready') update(i, { status: 'streaming' })
          if (e.type === 'token') setStates((prev) => prev.map((s, k) => (k === i ? { ...s, status: 'streaming', text: s.text + String(e.text) } : s)))
          if (e.type === 'done') update(i, { status: 'done', tokens: Number(e.tokens), tps: (e.tokensPerSecond as number) ?? null })
          if (e.type === 'error') update(i, { status: 'error', message: String(e.message) })
        })
      }
    } catch (err) {
      if (!ctrl.signal.aborted) setStates((prev) => prev.map((s) => (s.status === 'done' ? s : { ...s, status: 'error', message: `The connection to the backend stopped. ${(err as Error).message}` })))
    } finally {
      setRunning(false)
    }
  }

  const stop = () => {
    abort.current?.abort()
    setStates((prev) => prev.map((s) => (s.status === 'done' ? s : { ...s, status: 'idle' })))
    setRunning(false)
  }

  return (
    <div className="live-panel">
      <label className="ui" htmlFor="live-question"><strong>Ask your own question.</strong> Both models answer live on this computer's graphics card, one after the other.</label>
      <textarea id="live-question" value={question} maxLength={2000} onChange={(e) => setQuestion(e.target.value)}
        placeholder="For example: explain what a cache miss is in two sentences" style={{ marginTop: '0.6rem' }} />
      <div className="controls" style={{ marginTop: '0.75rem' }}>
        <button type="button" className="btn" onClick={ask} disabled={running || !question.trim()}>Ask both models</button>
        {running && <button type="button" className="btn quiet" onClick={stop}>Stop</button>}
      </div>
      <div className="answers two">
        {BEST_TWO.map((id, i) => {
          const m = modelById(results, id)
          return (
            <AnswerCard key={id} title={TITLES[id]} accent={FAMILY_COLOR[m.family]} meta={<ModelMeta model={m} speed={null} />}
              liveText={states[i].text} streaming={states[i].status === 'streaming'} foot={statusLine(states[i])} />
          )
        })}
      </div>
    </div>
  )
}

export default function TryBest() {
  const { results, gallery } = useData()
  const prompts = gallery.prompts
  const [index, setIndex] = useState(0)
  const [speedIndex, setSpeedIndex] = useState(1)
  const [runKey, setRunKey] = useState(0)
  const [live, setLive] = useState<'unknown' | 'checking' | 'on' | 'off'>('unknown')
  const prompt = prompts[index]
  const models = BEST_TWO.map((id) => modelById(results, id))
  const answers = models.map((m) => prompt.answers[m.id])
  const { counts, playing } = useReplay(answers, { mode: 'real', speed: SPEEDS[speedIndex].speed }, runKey)

  const check = useCallback(async () => {
    setLive('checking')
    setLive((await probe()) ? 'on' : 'off')
  }, [])

  // On the author's own machine the backend is looked for automatically; on the public site only after a click.
  useEffect(() => {
    if (['localhost', '127.0.0.1'].includes(window.location.hostname)) check()
  }, [check])

  return (
    <section id="try" className="section wrap" aria-labelledby="try-title">
      <div className="section-head">
        <h2 id="try-title">Try the two best 4 bit models</h2>
        <p className="lede">
          Both fit on one 8 GB graphics card. Pick a question and watch them answer at the speed they really ran, or faster.
        </p>
      </div>

      <div className="prompt-list" role="group" aria-label="Choose a question">
        {prompts.map((p, i) => (
          <button key={p.id} type="button" className="chip" aria-pressed={i === index} title={p.prompt}
            onClick={() => { setIndex(i); setRunKey((k) => k + 1) }}>
            {p.topic}: {p.prompt.split('\n')[0]}
          </button>
        ))}
      </div>

      <div className="controls">
        <button type="button" className="btn" onClick={() => setRunKey((k) => k + 1)} disabled={playing}>
          {playing ? 'Playing' : 'Play the answers'}
        </button>
        <div className="segmented" role="group" aria-label="Replay speed">
          {SPEEDS.map((s, i) => (
            <button key={s.label} type="button" aria-pressed={speedIndex === i} onClick={() => setSpeedIndex(i)}>{s.label}</button>
          ))}
        </div>
      </div>

      <div className="prompt-box"><span className="who">Question</span>{prompt.prompt}</div>

      <div className="answers two">
        {models.map((m, i) => (
          <AnswerCard
            key={m.id}
            title={TITLES[m.id]}
            accent={FAMILY_COLOR[m.family]}
            meta={<ModelMeta model={m} speed={answers[i]?.tokensPerSecond ?? null} />}
            answer={answers[i]}
            count={runKey === 0 ? 0 : counts[i]}
            streaming={playing && counts[i] < (answers[i]?.pieces.length ?? 0)}
            showDiff
            foot={runKey > 0 && !playing ? compareFoot(m.id, answers[i]) : m.note}
          />
        ))}
      </div>

      {live === 'on' ? <LivePanel /> : (
        <div className="live-panel">
          <p className="ui" style={{ margin: 0 }}>
            <span className="live-status"><span className="dot" aria-hidden="true" />
              {live === 'checking' ? 'Looking for a live backend on this computer.' : live === 'off'
                ? 'No live backend found on this computer.' : 'Live questions need the backend running on your own computer.'}</span>
          </p>
          <p className="note" style={{ margin: '0.6rem 0 0' }}>
            With a GPU of 8 GB or more, clone the repository and run <code>deployment/run_live.ps1</code>, then connect to ask both models anything.
          </p>
          <button type="button" className="btn quiet" style={{ marginTop: '0.9rem' }} onClick={check} disabled={live === 'checking'}>Connect to a local backend</button>
        </div>
      )}
    </section>
  )
}
