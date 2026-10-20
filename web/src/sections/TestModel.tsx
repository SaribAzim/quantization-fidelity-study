import { useCallback, useEffect, useRef, useState } from 'react'
import { DiffText, useReplay } from '../components/Replay'
import { BEST, modelById, PROJECT_PAGE, useData } from '../data'
import { num, pct, useSeenOnce } from '../lib'

const API = 'http://127.0.0.1:8000'
const SPEEDS = [{ label: 'Real speed', speed: 1 }, { label: '4 times', speed: 4 }, { label: '16 times', speed: 16 }]

type LiveState = { status: 'idle' | 'queued' | 'loading' | 'streaming' | 'done' | 'error'; text: string; tokens: number; tps: number | null; message?: string }
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

export default function TestModel() {
  const { results, gallery } = useData()
  const model = modelById(results, BEST)
  const original = modelById(results, 'original')
  const plain = modelById(results, results.baselineId)
  const prompts = gallery.prompts.filter((p) => p.answers[BEST])
  const speeds = prompts.map((p) => p.answers[BEST].tokensPerSecond ?? 0).filter((v) => v > 0)
  const avgSpeed = speeds.reduce((s, v) => s + v, 0) / Math.max(1, speeds.length)

  const [mode, setMode] = useState<'recorded' | 'live'>('recorded')
  const [index, setIndex] = useState(0)
  const [speedIndex, setSpeedIndex] = useState(1)
  const [runKey, setRunKey] = useState(0)
  const [showDiff, setShowDiff] = useState(false)
  const [live, setLive] = useState<'unknown' | 'checking' | 'on' | 'off'>('unknown')
  const [question, setQuestion] = useState('')
  const [liveQuestion, setLiveQuestion] = useState('')
  const [liveState, setLiveState] = useState<LiveState>(idle())
  const abort = useRef<AbortController | null>(null)
  const [panelRef, seen] = useSeenOnce<HTMLDivElement>(0.35)

  const prompt = prompts[index]
  const answer = prompt.answers[BEST]
  const { counts, playing } = useReplay([answer], { mode: 'real', speed: SPEEDS[speedIndex].speed }, runKey)
  const count = counts[0]

  // The first answer starts by itself when the tester comes into view.
  useEffect(() => { if (seen) setRunKey((k) => (k === 0 ? 1 : k)) }, [seen])

  const check = useCallback(async () => {
    setLive('checking')
    setLive((await probe()) ? 'on' : 'off')
  }, [])
  // On the author's own machine the backend is looked for automatically; on the public site only after a click.
  useEffect(() => {
    if (['localhost', '127.0.0.1'].includes(window.location.hostname)) check()
  }, [check])

  const pick = (i: number) => {
    abort.current?.abort()
    setMode('recorded')
    setIndex(i)
    setRunKey((k) => k + 1)
  }
  const setSpeed = (i: number) => { setSpeedIndex(i); if (playing) setRunKey((k) => k + 1) }

  const ask = async () => {
    const q = question.trim()
    if (!q) return
    abort.current?.abort()
    const ctrl = new AbortController()
    abort.current = ctrl
    setMode('live')
    setLiveQuestion(q)
    setQuestion('')
    setLiveState({ ...idle(), status: 'queued' })
    try {
      await streamAnswer(BEST, q, ctrl.signal, (e) => {
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
  const recordedLine = runKey === 0
    ? 'The answer starts when this panel comes into view.'
    : playing
      ? `${num(elapsed, 1)} s of recorded time, ${count} word pieces so far.`
      : `Recorded on an RTX 3060 Ti: ${answer.pieces.length} word pieces in ${num(answer.times[answer.times.length - 1] ?? 0, 1)} s. Same opening as the original for ${answer.identicalTokens} word pieces; ${Math.round(answer.wordSimilarity * 100)}% of the wording matches.`

  return (
    <section id="test" className="section wrap" aria-labelledby="test-title">
      <div className="section-head">
        <h2 id="test-title">Test the best 4 bit model</h2>
        <p className="lede">
          Groups of 128 with layers 0, 1 and 31 kept at 8 bit: the version closest to the original. Pick a question to watch its recorded answer
          at the speed it really ran, or faster. When the backend runs on your own computer, you can ask it anything.
        </p>
      </div>

      <div className="tester">
        <div className="console" ref={panelRef}>
          <div className="console-head">
            <span className={`dot${liveBusy || playing ? ' on' : ''}`} aria-hidden="true" />
            <strong>{model.label}</strong>
            <span className="console-mode">{mode === 'live' ? 'Live on this computer' : 'Recorded answer'}</span>
          </div>

          <div className="transcript">
            <div className="bubble q">
              <span className="who">Question</span>
              {mode === 'live' ? liveQuestion : prompt.prompt}
            </div>
            <div className="bubble a">
              <span className="who">4 bit model</span>
              <div className="answer-text" data-quote="model answer" aria-live="polite">
                {mode === 'live' ? liveState.text : <DiffText answer={answer} count={count} showDiff={showDiff} />}
                {(mode === 'live' ? liveState.status === 'streaming' : playing) && <span className="caret" aria-hidden="true" />}
              </div>
            </div>
            <p className="note console-status">{mode === 'live' ? liveLine(liveState) : recordedLine}</p>
          </div>

          <div className="composer">
            <label className="visually-hidden" htmlFor="ask">Your question</label>
            <textarea id="ask" value={question} maxLength={2000} rows={2} disabled={live !== 'on'}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!liveBusy) ask() } }}
              placeholder={live === 'on' ? 'Ask anything, for example: explain what a cache miss is in two sentences' : 'Typing your own question needs the backend on your computer. Pick a question below.'} />
            {liveBusy
              ? <button type="button" className="btn quiet" onClick={stop}>Stop</button>
              : <button type="button" className="btn" onClick={ask} disabled={live !== 'on' || !question.trim()}>Ask</button>}
          </div>

          <div className="console-foot">
            <div className="prompt-list" role="group" aria-label="Recorded questions">
              {prompts.map((p, i) => (
                <button key={p.id} type="button" className="chip" aria-pressed={mode === 'recorded' && i === index} title={p.prompt} onClick={() => pick(i)}>
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
              <label className="toggle ui">
                <input type="checkbox" checked={showDiff} onChange={(e) => setShowDiff(e.target.checked)} />
                Mark words that differ from the original
              </label>
            </div>
          </div>
        </div>

        <aside className="model-card" aria-label="About this model">
          <h3>Closest to the original</h3>
          <p className="note">{model.note}</p>
          <dl className="facts">
            <div><dt>Size</dt><dd>{model.metrics.sizeGb.toFixed(2)} GB <span className="fit">fits 8 GB</span></dd></div>
            <div><dt>Exam, 412 questions</dt><dd>{pct(model.metrics.exam)} <span className="vs">original {pct(original.metrics.exam)}</span></dd></div>
            <div><dt>Same next word as the original</dt><dd>{pct(model.metrics.top1)}</dd></div>
            <div><dt>Distance from the original</dt><dd>{num(model.metrics.kl ?? 0, 3)} <span className="vs">plain 4 bit {num(plain.metrics.kl ?? 0, 3)}</span></dd></div>
            <div><dt>Speed on an RTX 3060 Ti, average</dt><dd>{num(avgSpeed, 2)} word pieces a second</dd></div>
          </dl>
          <a href={`${PROJECT_PAGE}#study`}>How this model was made</a>

          <div className="live-box">
            <p className="live-status ui"><span className={`dot${live === 'on' ? ' on' : ''}`} aria-hidden="true" />
              {live === 'on' ? 'Live backend connected.' : live === 'checking' ? 'Looking for a live backend.' : live === 'off' ? 'No live backend found.' : 'Live questions run on your own computer.'}</p>
            {live !== 'on' && (
              <>
                <p className="note">With a GPU of 8 GB or more, clone the repository and run <code>deployment/run_live.ps1</code>, then connect.</p>
                <button type="button" className="btn quiet small" onClick={check} disabled={live === 'checking'}>Connect to a local backend</button>
              </>
            )}
          </div>
        </aside>
      </div>
    </section>
  )
}
