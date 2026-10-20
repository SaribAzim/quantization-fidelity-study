import { scaleLinear } from 'd3-scale'
import { FAMILY_COLOR, modelById, useData } from '../data'
import { pct, prefersReducedMotion, useSeenOnce } from '../lib'

const ROWS = [
  { id: 'original', name: 'Original, 16 bit' },
  { id: 'naive_8bit', name: 'Naive 8 bit' },
  { id: 'per_row_vkod', name: 'Best exam score, 4 bit' },
  { id: 'g128_l0131', name: 'Closest to the original, 4 bit' },
  { id: 'per_row', name: 'Plain 4 bit' },
]

function ExamDotPlot() {
  const { results } = useData()
  const [ref, seen] = useSeenOnce<SVGSVGElement>(0.4)
  const W = 560, H = 250, left = 210, right = 24, top = 18, rowH = 40
  const x = scaleLinear().domain([0.46, 0.6]).range([left, W - right])
  const ticks = [0.46, 0.5, 0.55, 0.6]
  const anim = !prefersReducedMotion()
  const originalExam = modelById(results, 'original').metrics.exam
  return (
    <svg ref={ref} className="dotplot" viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
      aria-label="Exam scores on 412 computer science questions, with their uncertainty, for five versions of the model.">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top - 6} y2={top + ROWS.length * rowH - 10} stroke="var(--rule)" />
          <text x={x(t)} y={top + ROWS.length * rowH + 8} textAnchor="middle">{pct(t, 0)}</text>
        </g>
      ))}
      <line x1={x(originalExam)} x2={x(originalExam)} y1={top - 6} y2={top + ROWS.length * rowH - 10} stroke="var(--blue)" strokeDasharray="3 4" />
      {ROWS.map((row, i) => {
        const m = modelById(results, row.id).metrics
        const color = FAMILY_COLOR[modelById(results, row.id).family]
        const y = top + i * rowH + 10
        const cx = seen || !anim ? x(m.exam) : x(0.46)
        return (
          <g key={row.id}>
            <text className="label" x={0} y={y + 5}>{row.name}</text>
            <line x1={x(m.exam - m.examStderr)} x2={x(m.exam + m.examStderr)} y1={y} y2={y} stroke={color} strokeWidth={2}
              style={{ opacity: seen || !anim ? 0.45 : 0, transition: 'opacity 600ms 700ms' }} />
            <circle cx={cx} cy={y} r={7} fill={color} style={{ transition: anim ? `cx 900ms cubic-bezier(.65,0,.35,1) ${i * 90}ms` : undefined }} />
            <text x={W - right} y={y + 5} textAnchor="end" style={{ opacity: seen || !anim ? 1 : 0, transition: 'opacity 400ms 900ms' }}>{pct(m.exam)}</text>
          </g>
        )
      })}
    </svg>
  )
}

export default function Question() {
  const { results } = useData()
  const original = modelById(results, 'original').metrics
  const plain = modelById(results, 'per_row').metrics
  const best = modelById(results, 'g128_l0131').metrics
  const bestExam = modelById(results, 'per_row_vkod').metrics
  const oneIn = Math.round(1 / (1 - plain.top1))
  return (
    <section className="section wrap" aria-labelledby="question-title">
      <div className="question-grid">
        <div className="section-head" style={{ marginBottom: 0 }}>
          <h2 id="question-title">How much does rounding cost, and what wins it back?</h2>
          <p>
            Every version of the model was measured against the original on the same text and the same {results.facts.examQuestions} computer science exam questions.
            The question is not only whether a small model still scores well, but how faithfully it copies the original.
          </p>
          <p>
            Plain 4 bit rounding drops the exam score from {pct(original.exam)} to {pct(plain.exam)}, and the model picks a different next word than the original about one time in {oneIn}.
            The best fixes bring the score back to {pct(bestExam.exam)} and cut the distance from the original's predictions from {plain.kl!.toFixed(3)} to {best.kl!.toFixed(3)}.
          </p>
          <p className="note">The exam has an uncertainty of about plus or minus {(results.facts.examNoise * 100).toFixed(1)} points, shown as the faint bars.</p>
        </div>
        <div className="panel" style={{ background: 'transparent', border: 0, padding: 0 }}>
          <ExamDotPlot />
        </div>
      </div>
    </section>
  )
}
