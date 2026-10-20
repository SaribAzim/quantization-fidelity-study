import { useEffect, useRef, useState, type ReactNode } from 'react'
import type { Answer } from '../data'
import { prefersReducedMotion } from '../lib'

export type Pace = { mode: 'shared'; piecesPerSecond: number } | { mode: 'real'; speed: number }

/**
 * Replays recorded answers. "shared" plays every answer at one pace so they can be compared side by side;
 * "real" uses each word piece's recorded time (optionally sped up). Returns how many pieces of each answer
 * are visible, and whether all are finished. A new runKey restarts the replay.
 */
export function useReplay(answers: (Answer | undefined)[], pace: Pace, runKey: number, startDelayMs = 0) {
  const [counts, setCounts] = useState<number[]>(() => answers.map(() => 0))
  const [playing, setPlaying] = useState(false)
  const frame = useRef(0)

  useEffect(() => {
    cancelAnimationFrame(frame.current)
    if (runKey === 0) { setCounts(answers.map(() => 0)); setPlaying(false); return }
    const totals = answers.map((a) => a?.pieces.length ?? 0)
    if (prefersReducedMotion()) { setCounts(totals); setPlaying(false); return }
    setCounts(answers.map(() => 0))
    setPlaying(true)
    const start = performance.now() + startDelayMs
    const tick = (now: number) => {
      const t = Math.max(0, (now - start) / 1000)
      const next = answers.map((a, i) => {
        if (!a) return 0
        if (pace.mode === 'shared') return Math.min(totals[i], Math.floor(t * pace.piecesPerSecond))
        const scaled = t * pace.speed
        let n = 0
        while (n < a.times.length && a.times[n] <= scaled) n++
        return n
      })
      setCounts(next)
      if (next.every((n, i) => n >= totals[i])) { setPlaying(false); return }
      frame.current = requestAnimationFrame(tick)
    }
    frame.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runKey])

  return { counts, playing }
}

/** Types a prompt out, character by character, when runKey changes. */
export function useTyping(text: string, runKey: number, charsPerSecond = 55) {
  const [shown, setShown] = useState(text.length)
  useEffect(() => {
    if (runKey === 0 || prefersReducedMotion()) { setShown(text.length); return }
    setShown(0)
    const start = performance.now()
    let frame = 0
    const tick = (now: number) => {
      const n = Math.min(text.length, Math.floor(((now - start) / 1000) * charsPerSecond))
      setShown(n)
      if (n < text.length) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [text, runKey, charsPerSecond])
  return text.slice(0, shown)
}

/** The visible part of an answer, with stretches that differ from the original highlighted. */
function DiffText({ answer, count, showDiff }: { answer: Answer; count: number; showDiff: boolean }) {
  const visible = answer.pieces.slice(0, count).join('').length
  const out: ReactNode[] = []
  let used = 0
  for (let i = 0; i < answer.diff.length && used < visible; i++) {
    const [text, same] = answer.diff[i]
    const part = text.slice(0, visible - used)
    used += part.length
    out.push(showDiff && !same ? <mark key={i}>{part}</mark> : <span key={i}>{part}</span>)
  }
  return <>{out}</>
}

type CardProps = {
  title: string
  accent: string
  meta: ReactNode
  answer?: Answer
  liveText?: string
  count?: number
  streaming: boolean
  showDiff?: boolean
  foot?: ReactNode
}

export function AnswerCard({ title, accent, meta, answer, liveText, count = 0, streaming, showDiff = true, foot }: CardProps) {
  return (
    <article className="answer" style={{ ['--accent' as never]: accent }}>
      <div className="answer-head">
        <h3>{title}</h3>
        <div className="answer-meta">{meta}</div>
      </div>
      <div className="answer-text" data-quote="model answer" aria-live="polite">
        {liveText !== undefined ? liveText : answer ? <DiffText answer={answer} count={count} showDiff={showDiff} /> : null}
        {streaming && <span className="caret" aria-hidden="true" />}
      </div>
      {!streaming && !liveText && (liveText !== undefined || (answer && count === 0)) && (
        <p className="answer-empty">{liveText !== undefined ? 'The live answer appears here.' : 'Press play to watch this answer appear.'}</p>
      )}
      {foot && <div className="answer-foot">{foot}</div>}
    </article>
  )
}
