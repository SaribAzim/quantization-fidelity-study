import { useMemo, useState } from 'react'
import { AnswerCard, useReplay, useTyping } from '../components/Replay'
import { FAMILY_COLOR, modelById, SHOWCASE, useData, type Model } from '../data'
import { pct } from '../lib'

const TITLES: Record<string, string> = {
  original: 'Original, 16 bit',
  naive_8bit: 'Naive 8 bit',
  g128_l0131: 'Closest to the original, 4 bit',
  per_row_vkod: 'Best exam score, 4 bit',
}
const SHARED_PACE = 22 // word pieces a second, the same for every column

export function ModelMeta({ model, speed }: { model: Model; speed: number | null }) {
  const m = model.metrics
  const fits = m.sizeGb <= 8
  return (
    <>
      {m.sizeGb.toFixed(1)} GB, <span className={fits ? 'fit' : 'nofit'}>{fits ? 'fits 8 GB' : 'too large for 8 GB'}</span>
      <br />
      Exam {pct(m.exam)}{speed ? <>, ran at {speed.toFixed(2)} word pieces a second</> : null}
    </>
  )
}

export function compareFoot(id: string, a?: { identicalTokens: number; wordSimilarity: number }) {
  if (!a) return 'No recorded answer for this question.'
  if (id === 'original') return 'The reference answer every other column is compared with.'
  return `Same opening as the original for ${a.identicalTokens} word pieces; ${Math.round(a.wordSimilarity * 100)}% of the wording matches.`
}

export default function Showcase() {
  const { results, gallery } = useData()
  const prompts = gallery.prompts
  const [index, setIndex] = useState(() => Math.max(0, prompts.findIndex((p) => p.id === 'prime_221')))
  const [runKey, setRunKey] = useState(0)
  const prompt = prompts[index]
  const models = SHOWCASE.map((id) => modelById(results, id))
  const answers = models.map((m) => prompt.answers[m.id])

  const typed = useTyping(prompt.prompt, runKey)
  const typingMs = (prompt.prompt.length / 55) * 1000 + 250
  const { counts, playing } = useReplay(answers, { mode: 'shared', piecesPerSecond: SHARED_PACE }, runKey, typingMs)
  const finished = runKey > 0 && !playing

  const slow = useMemo(() => {
    const o = gallery.prompts.flatMap((p) => p.answers.original ? [p.answers.original.tokensPerSecond ?? 0] : [])
    const b = gallery.prompts.flatMap((p) => p.answers.g128_l0131 ? [p.answers.g128_l0131.tokensPerSecond ?? 0] : [])
    const avg = (xs: number[]) => xs.reduce((s, v) => s + v, 0) / Math.max(1, xs.length)
    return Math.round(avg(b) / Math.max(0.01, avg(o)))
  }, [gallery])

  const play = (i?: number) => { if (i !== undefined) setIndex(i); setRunKey((k) => k + 1) }

  return (
    <section id="answers" className="section wrap" aria-labelledby="answers-title">
      <div className="section-head">
        <h2 id="answers-title">Same question, four models</h2>
        <p className="lede">
          Real answers, recorded on one graphics card. The original and the 8 bit model are too large for 8 GB, so they ran from system memory,
          about {slow} times slower than the 4 bit models. Here every answer replays at one shared pace; each column shows how fast that model really ran.
        </p>
      </div>

      <div className="prompt-list" role="group" aria-label="Choose a question">
        {prompts.map((p, i) => (
          <button key={p.id} type="button" className="chip" aria-pressed={i === index} onClick={() => play(i)} title={p.prompt}>
            {p.topic}: {p.prompt.split('\n')[0]}
          </button>
        ))}
      </div>

      <div className="prompt-box">
        <span className="who">Question</span>
        {runKey > 0 ? typed : prompt.prompt}
      </div>

      <div className="controls">
        <button type="button" className="btn" onClick={() => play()} disabled={playing}>
          {runKey === 0 ? 'Play the answers' : playing ? 'Playing' : 'Play again'}
        </button>
        <span className="note"><span className="legend-mark">Highlighted words</span> differ from the original's answer. Answers stop at 80 word pieces.</span>
      </div>

      <div className="answers four">
        {models.map((m, i) => (
          <AnswerCard
            key={m.id}
            title={TITLES[m.id]}
            accent={FAMILY_COLOR[m.family]}
            meta={<ModelMeta model={m} speed={answers[i]?.tokensPerSecond ?? null} />}
            answer={answers[i]}
            count={counts[i]}
            streaming={playing && answers[i] !== undefined && counts[i] < answers[i]!.pieces.length}
            showDiff={finished || playing}
            foot={finished ? compareFoot(m.id, answers[i]) : undefined}
          />
        ))}
      </div>
    </section>
  )
}
