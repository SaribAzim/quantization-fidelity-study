import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { DiffText, useReplay } from '../components/Replay'
import { averageSpeed, BEST, FAMILY_COLOR, PROJECT_PAGE, useData } from '../data'
import { num, pct, useSeenOnce } from '../lib'

const API = 'http://127.0.0.1:8000'
const SPEEDS = [{ label: 'Real speed', speed: 1 }, { label: '4 times', speed: 4 }, { label: '16 times', speed: 16 }]

type LiveState = { status: 'idle' | 'queued' | 'loading' | 'streaming' | 'done' | 'error'; text: string; tokens: number; tps: number | null; message?: string }
const idle = (): LiveState => ({ status: 'idle', text: '', tokens: 0, tps: null })

/** Asks the local backend (deployment/server.py) which models it can run. Never called automatically on the public site. */
async function probe(): Promise<string[] | null> {
  try {
    const ctrl = new AbortController()
    const t = window.setTimeout(() => ctrl.abort(), 1500)
    const r = await fetch(`${API}/api/models`, { signal: ctrl.signal })
    window.clearTimeout(t)
    if (!r.ok) return null
    return ((await r.json()) as { id: string }[]).map((m) => m.id)
  } catch {
    return null
  }
}

async function streamAnswer(model: string, prompt: string, signal: AbortSignal, onEvent: (e: Record<string, unknown>) => void) {
  const r = await fetch(`${API}/api/generate`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, signal,
    body: JSON.stringify({ model, prompt, maxNewTokens: 160 }),
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

function liveLine(s: LiveState): string {
  switch (s.status) {
    case 'queued': return 'Waiting for the graphics card.'
    case 'loading': return 'Loading the model onto the graphics card, about 20 seconds.'
    case 'streaming': return 'Answering live.'
    case 'done': return `Finished: ${s.tokens} word pieces at ${s.tps ?? 0} a second, live on this computer.`
    case 'error': return s.message ?? 'Something went wrong.'
    default: return ''
  }
}

export default function Workbench() {
  const { results, gallery } = useData()
  // Every model with recorded answers, best 4 bit first, then by distance from the original.
  const models = useMemo(() => results.models
    .filter((m) => gallery.prompts.some((p) => p.answers[m.id]))
    .sort((a, b) => (a.id === BEST ? -1 : b.id === BEST ? 1 : (a.metrics.kl ?? -1) - (b.metrics.kl ?? -1))), [results, gallery])
  const original = results.models.find((m) => m.id === 'original')!

  const [modelId, setModelId] = useState(BEST)
  const model = models.find((m) => m.id === modelId) ?? models[0]
  const prompts = gallery.prompts.filter((p) => p.answers[model.id])

  const [mode, setMode] = useState<'recorded' | 'live'>('recorded')
  const [index, setIndex] = useState(0)
  const [speedIndex, setSpeedIndex] = useState(1)
  const [runKey, setRunKey] = useState(0)
  const [showDiff, setShowDiff] = useState(false)
  const [live, setLive] = useState<'unknown' | 'checking' | 'off' | string[]>('unknown')
  const [question, setQuestion] = useState('')
  const [liveQuestion, setLiveQuestion] = useState('')
  const [liveState, setLiveState] = useState<LiveState>(idle())
  const abort = useRef<AbortController | null>(null)
  const [panelRef, seen] = useSeenOnce<HTMLDivElement>(0.35)

  const prompt = prompts[Math.min(index, prompts.length - 1)]
  const answer = prompt.answers[model.id]
  const { counts, playing } = useReplay([answer], { mode: 'real', speed: SPEEDS[speedIndex].speed }, runKey)
  const count = counts[0]

  useEffect(() => { if (seen) setRunKey((k) => (k === 0 ? 1 : k)) }, [seen])

  const check = useCallback(async () => {
    setLive('checking')
    setLive((await probe()) ?? 'off')
  }, [])
  // On the author's own machine the backend is looked for automatically; on the public site only after a click.
  useEffect(() => {
    if (['localhost', '127.0.0.1'].includes(window.location.hostname)) check()
  }, [check])

  const connected = Array.isArray(live)
  const canRunLive = connected && live.includes(model.id)

  const chooseModel = (id: string) => {
    abort.current?.abort()
    setModelId(id)
    setMode('recorded')
    setRunKey((k) => k + 1)
  }
  const pick = (i: number) => {
    abort.current?.abort()
    setMode('recorded')
    setIndex(i)
    setRunKey((k) => k + 1)
  }
  const setSpeed = (i: number) => { setSpeedIndex(i); if (playing) setRunKey((k) => k + 1) }

  const ask = async () => {
    const q = question.trim()
    if (!q || !canRunLive) return
    abort.current?.abort()
    const ctrl = new AbortController()
    abort.current = ctrl
    setMode('live')
    setLiveQuestion(q)
    setQuestion('')
    setLiveState({ ...idle(), status: 'queued' })
    try {
      await streamAnswer(model.id, q, ctrl.signal, (e) => {
        if (e.type === 'loading') setLiveState((s) => ({ ...s, status: 'loading' }))
        if (e.type === 'ready') setLiveState((s) => ({ ...s, status: 'streaming' }))
        if (e.type === 'token') setLiveState((s) => ({ ...s, status: 'streaming', text: s.text + String(e.text) }))
        if (e.type === 'done') setLiveState((s) => ({ ...s, status: 'done', tokens: Number(e.tokens), tps: (e.tokensPerSecond as number) ?? null }))
        if (e.type === 'error') setLiveState((s) => ({ ...s, status: 'error', message: String(e.message) }))
      })
    } catch (err) {
      if (!ctrl.signal.aborted) setLiveState((s) => ({ ...s, status: 'error', message: `The connection to the backend stopped. ${(err as Error).message}` }))
    }
  }
  const stop = () => { abort.current?.abort(); setLiveState((s) => ({ ...s, status: s.text ? 'done' : 'idle' })) }

  const liveBusy = mode === 'live' && ['queued', 'loading', 'streaming'].includes(liveState.status)
  const elapsed = count > 0 ? answer.times[count - 1] : 0
  const isRef = model.id === 'original'
  const recordedLine = runKey === 0
    ? 'The answer starts when this panel comes into view.'
    : playing
      ? `${num(elapsed, 1)} s of recorded time, ${count} word pieces so far.`
      : `Recorded on an RTX 3060 Ti: ${answer.pieces.length} word pieces in ${num(answer.times[answer.times.length - 1] ?? 0, 1)} s.`
        + (isRef ? '' : ` Same opening as the original for ${answer.identicalTokens} word pieces; ${Math.round(answer.wordSimilarity * 100)}% of the wording matches.`)
  const placeholder = !connected
    ? 'Live questions need the backend on this computer. Pick a recorded question below.'
    : canRunLive ? 'Ask anything, for example: explain what a cache miss is in two sentences'
      : 'This model is too large to run live on an 8 GB card. Choose a 4 bit model to ask your own question.'
  const speed = averageSpeed(gallery, model.id)

  return (
    <section className="section wrap page-top" aria-labelledby="bench-title">
      <div className="section-head">
        <h1 id="bench-title" className="page-title">Workbench</h1>
        <p className="lede">
          The testing bench for every quantized model in the study. Replay any model's recorded answers, or, with the backend running on this computer,
          ask the 4 bit models your own questions live.
        </p>
      </div>

      <div className="bench">
        <aside className="bench-side">
          <h2 className="side-title">Models</h2>
          <ul className="model-list">
            {models.map((m) => (
              <li key={m.id}>
                <button type="button" className="model-item" aria-pressed={m.id === model.id} onClick={() => chooseModel(m.id)}
                  style={{ ['--accent' as never]: FAMILY_COLOR[m.family] }}>
                  <span className="model-name">{m.label}</span>
                  <span className="model-meta">{m.metrics.sizeGb.toFixed(1)} GB, exam {pct(m.metrics.exam)}
                    {connected && <>, {live.includes(m.id) ? 'live ready' : 'recorded only'}</>}</span>
                </button>
              </li>
            ))}
          </ul>

          <div className="live-box">
            <p className="live-status ui"><span className={`dot${connected ? ' on' : ''}`} aria-hidden="true" />
              {connected ? `Live backend connected: ${live.length} models can run here.` : live === 'checking' ? 'Looking for a live backend.' : live === 'off' ? 'No live backend found.' : 'Live questions run on this computer.'}</p>
            {!connected && (
              <>
                <p className="note">
                  Start it from the project folder with <code>deployment/run_live.ps1</code>. It needs a graphics card with 8 GB and the quantized model folders,
                  which this project's scripts build from Meta's weights; they are not stored in the repository.
                </p>
                <button type="button" className="btn quiet small" onClick={check} disabled={live === 'checking'}>Connect to a local backend</button>
              </>
            )}
          </div>
        </aside>

        <div className="console" ref={panelRef}>
          <div className="console-head" style={{ ['--accent' as never]: FAMILY_COLOR[model.family] }}>
            <span className={`dot${liveBusy || playing ? ' on' : ''}`} aria-hidden="true" />
            <strong>{model.label}</strong>
            <span className="console-mode">{mode === 'live' ? 'Live on this computer' : 'Recorded answer'}</span>
          </div>
          <dl className="console-facts">
            <div><dt>Size</dt><dd>{model.metrics.sizeGb.toFixed(2)} GB</dd></div>
            <div><dt>Exam</dt><dd>{pct(model.metrics.exam)}</dd></div>
            <div><dt>Same next word</dt><dd>{isRef ? 'Reference' : pct(model.metrics.top1)}</dd></div>
            <div><dt>Distance</dt><dd>{model.metrics.kl === null ? 'Reference' : num(model.metrics.kl, 3)}</dd></div>
            <div><dt>Recorded speed</dt><dd>{speed ? `${num(speed, 2)} a second` : 'None'}</dd></div>
          </dl>

          <div className="transcript">
            <div className="bubble q">
              <span className="who">Question</span>
              {mode === 'live' ? liveQuestion : prompt.prompt}
            </div>
            <div className="bubble a" style={{ ['--accent' as never]: FAMILY_COLOR[model.family] }}>
              <span className="who">{model.label}</span>
              <div className="answer-text" data-quote="model answer" aria-live="polite">
                {mode === 'live' ? liveState.text : <DiffText answer={answer} count={count} showDiff={showDiff && !isRef} />}
                {(mode === 'live' ? liveState.status === 'streaming' : playing) && <span className="caret" aria-hidden="true" />}
              </div>
            </div>
            <p className="note console-status">{mode === 'live' ? liveLine(liveState) : recordedLine}</p>
          </div>

          <div className="composer">
            <label className="visually-hidden" htmlFor="ask">Your question</label>
            <textarea id="ask" value={question} maxLength={2000} rows={2} disabled={!canRunLive}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!liveBusy) ask() } }}
              placeholder={placeholder} />
            {liveBusy
              ? <button type="button" className="btn quiet" onClick={stop}>Stop</button>
              : <button type="button" className="btn" onClick={ask} disabled={!canRunLive || !question.trim()}>Ask</button>}
          </div>

          <div className="console-foot">
            <div className="prompt-list" role="group" aria-label="Recorded questions">
              {prompts.map((p, i) => (
                <button key={p.id} type="button" className="chip" aria-pressed={mode === 'recorded' && p.id === prompt.id} title={p.prompt} onClick={() => pick(i)}>
                  {p.topic}: {p.prompt.split('\n')[0]}
                </button>
              ))}
            </div>
            <div className="controls">
              <div className="segmented" role="group" aria-label="Replay speed">
                {SPEEDS.map((s, i) => (
                  <button key={s.label} type="button" aria-pressed={speedIndex === i} onClick={() => setSpeed(i)}>{s.label}</button>
                ))}
              </div>
              <button type="button" className="btn quiet" onClick={() => { setMode('recorded'); setRunKey((k) => k + 1) }} disabled={playing}>Play again</button>
              {!isRef && (
                <label className="toggle ui">
                  <input type="checkbox" checked={showDiff} onChange={(e) => setShowDiff(e.target.checked)} />
                  Mark words that differ from the original
                </label>
              )}
            </div>
          </div>
        </div>
      </div>
      <p className="note bench-foot">Original exam score for comparison: {pct(original.metrics.exam)}. <a href={`${PROJECT_PAGE}#study`}>How the models were made</a></p>
    </section>
  )
}
