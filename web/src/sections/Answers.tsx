import { useEffect, useMemo, useRef, useState } from 'react'
import { DiffText, pieceAgreement, useReplay, useTyping } from '../components/Replay'
import { averageSpeed, modelById, SHOWCASE, useData, type Answer, type Model } from '../data'
import { num, pct, useSeenOnce } from '../lib'

const PROFILES: Record<string, { title: string; sub: string; accent: string }> = {
  original: { title: 'Original', sub: '16 bit, as Meta released it.', accent: 'var(--blue)' },
  naive_8bit: { title: '8 bit', sub: 'Every weight rounded to one of 256 values.', accent: 'var(--slate)' },
  g128_l0131: { title: 'Best 4 bit', sub: '16 values per group of 128 weights, with layers 0, 1 and 31 kept at 8 bit.', accent: 'var(--sage)' },
}
const SHARED_PACE = 22 // word pieces a second, the same for every column
const GPU_GB = 8

/* Model profiles: the same four small charts for each model, on shared scales so the cards compare. */

type MeterProps = { label: string; value: string; frac: number; color: string; tick?: number; note?: string; seen: boolean; delay: number }
function Meter({ label, value, frac, color, tick, note, seen, delay }: MeterProps) {
  return (
    <div className="meter">
      <div className="meter-head"><span>{label}</span><strong>{value}</strong></div>
      <div className="meter-track" aria-hidden="true">
        <div className="meter-fill" style={{ width: seen ? `${Math.max(0, Math.min(1, frac)) * 100}%` : 0, background: color, transitionDelay: `${delay}ms` }} />
        {tick !== undefined && <div className="meter-tick" style={{ left: `${tick * 100}%` }} />}
      </div>
      {note && <div className="meter-note">{note}</div>}
    </div>
  )
}

function Profiles() {
  const { results, gallery } = useData()
  const [ref, seen] = useSeenOnce<HTMLDivElement>(0.25)
  const models = SHOWCASE.map((id) => modelById(results, id))
  const original = models[0]
  const speeds = models.map((m) => averageSpeed(gallery, m.id))
  const maxSpeed = Math.max(...speeds.map((s) => s ?? 0)) * 1.15 || 1
  const EXAM_LO = 0.4, EXAM_HI = 0.6

  return (
    <div className="profiles" ref={ref}>
      {models.map((m, i) => {
        const p = PROFILES[m.id]
        const fits = m.metrics.sizeGb <= GPU_GB
        const isRef = m.id === 'original'
        const d = i * 120
        return (
          <article key={m.id} className="profile" style={{ ['--accent' as never]: p.accent }}>
            <div className="profile-head">
              <h3>{p.title}</h3>
              <span className={`badge ${fits ? 'fit' : 'nofit'}`}>{fits ? 'Fits 8 GB' : 'Too large for 8 GB'}</span>
            </div>
            <p className="profile-sub">{p.sub}</p>
            <Meter label="Size" value={`${m.metrics.sizeGb.toFixed(1)} GB`} frac={m.metrics.sizeGb / 16} tick={0.5}
              color={fits ? 'var(--sage)' : 'var(--ember)'} note="Line: one 8 GB graphics card" seen={seen} delay={d} />
            <Meter label="Exam score" value={pct(m.metrics.exam)} frac={(m.metrics.exam - EXAM_LO) / (EXAM_HI - EXAM_LO)}
              tick={(original.metrics.exam - EXAM_LO) / (EXAM_HI - EXAM_LO)} color={p.accent}
              note={isRef ? '412 computer science questions' : `Line: the original, ${pct(original.metrics.exam)}`} seen={seen} delay={d + 80} />
            <Meter label="Same next word as the original" value={isRef ? 'Reference' : pct(m.metrics.top1)} frac={m.metrics.top1} color={p.accent}
              note={isRef ? 'Every other model is compared with it' : 'Over thousands of predictions'} seen={seen} delay={d + 160} />
            <Meter label="Speed on an RTX 3060 Ti" value={speeds[i] ? `${num(speeds[i]!, 2)} a second` : 'Recording'} frac={(speeds[i] ?? 0) / maxSpeed}
              color={p.accent} note={fits ? 'Word pieces, fully on the graphics card' : 'Word pieces, partly from system memory'} seen={seen} delay={d + 240} />
          </article>
        )
      })}
    </div>
  )
}

/* The replay: one column per model, with a strip that fills as the answer streams. */

function Strip({ total, count, agreement, isRef }: { total: number; count: number; agreement: boolean[]; isRef: boolean }) {
  const first = isRef ? -1 : agreement.indexOf(false)
  return (
    <div className="strip-wrap">
      <div className="strip" aria-hidden="true">
        {Array.from({ length: total }, (_, k) => (
          <span key={k} className="cell" data-state={k >= count ? 'empty' : isRef ? 'ref' : agreement[k] ? 'same' : 'diff'} />
        ))}
      </div>
      <div className="strip-note">
        {isRef ? 'Reference answer'
          : count === 0 ? 'Waiting'
            : first >= 0 && count > first ? `First difference at word piece ${first + 1}`
              : 'Matching the original so far'}
      </div>
    </div>
  )
}

function Column({ model, answer, count, playing, finished }: { model: Model; answer?: Answer; count: number; playing: boolean; finished: boolean }) {
  const p = PROFILES[model.id]
  const isRef = model.id === 'original'
  const agreement = useMemo(() => (answer ? pieceAgreement(answer) : []), [answer])
  const total = answer?.pieces.length ?? 0
  const streaming = playing && count < total
  return (
    <article className="column" style={{ ['--accent' as never]: p.accent }}>
      <header className="column-head">
        <h3>{p.title}</h3>
        <span className="column-speed">{answer?.tokensPerSecond ? `Ran at ${num(answer.tokensPerSecond, 2)} word pieces a second` : ''}</span>
      </header>
      {answer ? (
        <>
          <Strip total={total} count={count} agreement={agreement} isRef={isRef} />
          <div className="answer-text" data-quote="model answer" aria-live="polite">
            <DiffText answer={answer} count={count} showDiff={!isRef} />
            {streaming && <span className="caret" aria-hidden="true" />}
          </div>
          <footer className="column-foot">
            {finished
              ? isRef ? 'The answer the other two are compared with.'
                : `Same wording as the original for the first ${answer.identicalTokens} word pieces; ${Math.round(answer.wordSimilarity * 100)}% of its words match overall.`
              : `${count} of ${total} word pieces`}
          </footer>
        </>
      ) : (
        <p className="column-empty">This answer is still being recorded. It appears here once the recording finishes.</p>
      )}
    </article>
  )
}

/* Across all questions: word similarity to the original for the 8 bit and best 4 bit models. */

function Across({ onPick }: { onPick: (i: number) => void }) {
  const { gallery } = useData()
  const [ref, seen] = useSeenOnce<HTMLDivElement>(0.2)
  const others = SHOWCASE.filter((id) => id !== 'original')
  const avg = (id: string, key: 'wordSimilarity' | 'identicalTokens') => {
    const v = gallery.prompts.map((p) => p.answers[id]?.[key]).filter((x): x is number => x !== undefined)
    return v.length ? v.reduce((s, x) => s + x, 0) / v.length : null
  }
  const summary = others.map((id) => ({ id, sim: avg(id, 'wordSimilarity'), open: avg(id, 'identicalTokens') }))

  return (
    <div className="across" ref={ref}>
      <div className="across-legend ui">
        {others.map((id) => (
          <span key={id}><span className="key" style={{ background: PROFILES[id].accent }} />{PROFILES[id].title}</span>
        ))}
        <span className="muted">0% shares no wording with the original; 100% is word for word identical.</span>
      </div>
      <ul className="across-rows">
        <li className="across-row avg">
          <span className="across-label">Average of all questions</span>
          <span className="across-track">
            {summary.map((s, k) => s.sim !== null && (
              <span key={s.id} className="across-dot big" style={{ left: seen ? `${s.sim * 100}%` : '0%', background: PROFILES[s.id].accent, transitionDelay: `${k * 100}ms` }}
                title={`${PROFILES[s.id].title}: ${Math.round(s.sim * 100)}%`} />
            ))}
          </span>
          <span className="across-values">{summary.map((s) => (s.sim !== null ? `${Math.round(s.sim * 100)}%` : '')).filter(Boolean).join(', ')}</span>
        </li>
        {gallery.prompts.map((p, i) => (
          <li key={p.id}><button type="button" className="across-row" onClick={() => onPick(i)} title={`Replay: ${p.prompt}`}>
            <span className="across-label">{p.topic}: {p.prompt.split('\n')[0]}</span>
            <span className="across-track">
              {others.map((id, k) => {
                const a = p.answers[id]
                return a ? <span key={id} className="across-dot" style={{ left: seen ? `${a.wordSimilarity * 100}%` : '0%', background: PROFILES[id].accent, transitionDelay: `${150 + i * 50 + k * 60}ms` }} /> : null
              })}
            </span>
            <span className="across-values">{others.map((id) => (p.answers[id] ? `${Math.round(p.answers[id].wordSimilarity * 100)}%` : '')).filter(Boolean).join(', ')}</span>
          </button></li>
        ))}
      </ul>
      <div className="across-axis" aria-hidden="true"><span>0%</span><span>50%</span><span>100%</span></div>
      <p className="note across-sum">
        {summary.map((s) => s.sim !== null && s.open !== null
          ? `${PROFILES[s.id].title}: keeps the original's exact wording for the first ${num(s.open, 0)} word pieces on average, and ${Math.round(s.sim * 100)}% of its words match. `
          : `${PROFILES[s.id].title}: answers are still being recorded. `)}
      </p>
    </div>
  )
}

export default function Answers() {
  const { results, gallery } = useData()
  const prompts = gallery.prompts
  const [index, setIndex] = useState(() => Math.max(0, prompts.findIndex((p) => p.id === 'prime_221')))
  const [runKey, setRunKey] = useState(0)
  const stageRef = useRef<HTMLElement>(null)
  const prompt = prompts[index]
  const models = SHOWCASE.map((id) => modelById(results, id))
  const answers = models.map((m) => prompt.answers[m.id])

  const typed = useTyping(prompt.prompt, runKey)
  const typingMs = (prompt.prompt.length / 55) * 1000 + 250
  const { counts, playing } = useReplay(answers, { mode: 'shared', piecesPerSecond: SHARED_PACE }, runKey, typingMs)
  const finished = runKey > 0 && !playing

  const play = (i?: number) => { if (i !== undefined) setIndex(i); setRunKey((k) => k + 1) }
  const pickFromChart = (i: number) => { play(i); stageRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }) }

  // The page opens with the question typing itself and the three answers streaming in.
  useEffect(() => {
    const t = window.setTimeout(() => setRunKey((k) => (k === 0 ? 1 : k)), 900)
    return () => window.clearTimeout(t)
  }, [])

  return (
    <>
      <section className="section wrap page-top" aria-labelledby="answers-title">
        <div className="section-head">
          <h1 id="answers-title" className="page-title">Watch the models answer</h1>
          <p className="lede">One question, three versions of the same model. Every answer here is real, recorded on one graphics card, and replayed word by word.</p>
        </div>
        <Profiles />
      </section>

      <section ref={stageRef} id="replay" className="section wrap stage" aria-labelledby="replay-title">
        <div className="stage-head">
          <h2 id="replay-title">Same question, side by side</h2>
          <div className="prompt-list" role="group" aria-label="Choose a question">
            {prompts.map((p, i) => (
              <button key={p.id} type="button" className="chip" aria-pressed={i === index} onClick={() => play(i)} title={p.prompt}>
                {p.topic}: {p.prompt.split('\n')[0]}
              </button>
            ))}
          </div>
        </div>

        <div className="prompt-box">
          <span className="who">Question</span>
          {runKey > 0 ? typed : prompt.prompt}
          {runKey > 0 && typed.length < prompt.prompt.length && <span className="caret" aria-hidden="true" />}
        </div>

        <div className="controls">
          <button type="button" className="btn" onClick={() => play()} disabled={playing}>{playing ? 'Playing' : 'Play again'}</button>
        </div>

        <div className="columns">
          {models.map((m, i) => (
            <Column key={m.id} model={m} answer={answers[i]} count={counts[i]} playing={playing} finished={finished} />
          ))}
        </div>

        <ul className="howto">
          <li><span className="howto-sample"><mark>words</mark></span><span><strong>Highlighted words</strong> are where an answer's wording differs from the original's.</span></li>
          <li><span className="howto-sample strip-sample" aria-hidden="true">{[1, 1, 1, 1, 0, 1, 0, 0].map((s, k) => <span key={k} className="cell" data-state={s ? 'same' : 'diff'} />)}</span>
            <span><strong>The strip</strong> fills one cell per word piece: green while it matches the original, gold where it drifts.</span></li>
          <li><span className="howto-sample pace" aria-hidden="true"><span /><span /><span /></span>
            <span><strong>One shared pace</strong> for a fair comparison. Each column states how fast that model really ran.</span></li>
        </ul>
      </section>

      <section className="section wrap" aria-labelledby="across-title">
        <div className="section-head">
          <h2 id="across-title">Across all ten questions</h2>
          <p className="lede">How much of each answer's wording matches the original's. Select a question to replay it.</p>
        </div>
        <Across onPick={pickFromChart} />
      </section>
    </>
  )
}
