import { useMemo, useRef, useState } from 'react'
import { scaleLinear } from 'd3-scale'
import { FAMILY_COLOR, useData, type Model } from '../data'
import { pct, prefersReducedMotion, signedPct, useSeenOnce } from '../lib'

type Col = { key: string; label: string; value: (m: Model) => number | null; format: (v: number) => string; better: 'high' | 'low' | null }

const COLS: Col[] = [
  { key: 'size', label: 'Size', value: (m) => m.metrics.sizeGb, format: (v) => `${v.toFixed(2)} GB`, better: 'low' },
  { key: 'kl', label: 'Distance from the original', value: (m) => m.metrics.kl, format: (v) => v.toFixed(3), better: 'low' },
  { key: 'top1', label: 'Same next word', value: (m) => (m.id === 'original' ? null : m.metrics.top1), format: (v) => pct(v), better: 'high' },
  { key: 'exam', label: 'Exam score', value: (m) => m.metrics.exam, format: (v) => pct(v), better: 'high' },
  { key: 'ppl', label: 'Perplexity', value: (m) => m.metrics.perplexity, format: (v) => v.toFixed(2), better: 'low' },
]

function Leaderboard() {
  const { results } = useData()
  const [sort, setSort] = useState<{ key: string; dir: 'ascending' | 'descending' }>({ key: 'kl', dir: 'ascending' })
  const models = results.models

  const stats = useMemo(() => Object.fromEntries(COLS.map((c) => {
    const vals = models.map(c.value).filter((v): v is number => v !== null)
    const best = c.better === 'low' ? Math.min(...vals) : Math.max(...vals)
    return [c.key, { lo: Math.min(...vals), hi: Math.max(...vals), best }]
  })), [models])

  const rows = useMemo(() => {
    const col = COLS.find((c) => c.key === sort.key)!
    return [...models].sort((a, b) => {
      const va = col.value(a), vb = col.value(b)
      if (va === null) return -1
      if (vb === null) return 1
      return sort.dir === 'ascending' ? va - vb : vb - va
    })
  }, [models, sort])

  const toggle = (key: string) => setSort((s) => (s.key === key ? { key, dir: s.dir === 'ascending' ? 'descending' : 'ascending' } : { key, dir: COLS.find((c) => c.key === key)!.better === 'high' ? 'descending' : 'ascending' }))

  return (
    <div className="table-wrap">
      <table className="board">
        <caption className="visually-hidden">Every model measured against the original. Select a column heading to sort.</caption>
        <thead>
          <tr>
            <th scope="col">Model</th>
            {COLS.map((c) => (
              <th key={c.key} scope="col" aria-sort={sort.key === c.key ? sort.dir : undefined}>
                <button type="button" onClick={() => toggle(c.key)}>{c.label}</button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((m) => (
            <tr key={m.id}>
              <td>
                <span className="swatch" style={{ background: FAMILY_COLOR[m.family] }} aria-hidden="true" />
                <span className="model-name">{m.label}</span>
                {m.featured && <span className="tag">{m.featured}</span>}
                {m.id === 'original' && <span className="tag" style={{ color: 'var(--blue)' }}>The reference</span>}
              </td>
              {COLS.map((c) => {
                const v = c.value(m)
                if (v === null) return <td key={c.key} className="note">reference</td>
                const s = stats[c.key]
                const t = s.hi > s.lo ? (v - s.lo) / (s.hi - s.lo) : 0.5
                const goodness = c.better === 'low' ? 1 - t : t
                const shade = c.key === 'size' ? 0 : goodness
                return (
                  <td key={c.key} className={`shade${v === s.best && c.key !== 'size' ? ' best' : ''}`} style={{ ['--s' as never]: shade.toFixed(3) }}>
                    {c.format(v)}{c.key === 'size' && !m.metrics.sizeMeasured ? '*' : ''}{c.key === 'ppl' && m.id === 'qlora' ? '†' : ''}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function RecoveryChart() {
  const { results } = useData()
  const [ref, seen] = useSeenOnce<SVGSVGElement>(0.3)
  const rows = results.models
    .filter((m) => m.recovery && m.id !== 'naive_8bit' && m.recovery.kl !== null)
    .sort((a, b) => (b.recovery!.kl ?? 0) - (a.recovery!.kl ?? 0))
  // Each row: the model's name on its own line, then its two bars, so long names are never cut.
  const W = 560, rowH = 48, top = 34, gap = 24, H = top + rows.length * rowH + 4
  const colW = (W - gap) / 2
  const xs = [0, colW + gap]
  const scale = (v: number) => Math.max(-1, Math.min(1.1, v)) * (colW * 0.3)
  const anim = !prefersReducedMotion()
  return (
    <svg ref={ref} className="chart" viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
      aria-label="How much of the plain 4 bit model's loss each technique recovers, on distance from the original and on the exam.">
      {['Distance recovered', 'Exam points recovered'].map((t, k) => (
        <g key={t}>
          <text className="strong" x={xs[k] + colW / 2} y={14} textAnchor="middle">{t}</text>
          <line className="gridline" x1={xs[k] + colW / 2} x2={xs[k] + colW / 2} y1={top - 10} y2={H - 4} />
        </g>
      ))}
      {rows.map((m, i) => {
        const y = top + i * rowH
        return (
          <g key={m.id}>
            <text className="strong" x={0} y={y + 10}>{m.label}</text>
            {[m.recovery!.kl ?? 0, m.recovery!.exam].map((v, k) => {
              const zero = xs[k] + colW / 2
              const w = scale(v)
              const width = seen || !anim ? Math.abs(w) : 0
              return (
                <g key={k}>
                  <rect x={w >= 0 ? zero : zero - width} y={y + 18} width={width} height={16} rx={3}
                    fill={v >= 0 ? 'var(--teal)' : 'var(--ember)'}
                    style={{ transition: anim ? `width 800ms cubic-bezier(.65,0,.35,1) ${i * 60}ms, x 800ms cubic-bezier(.65,0,.35,1) ${i * 60}ms` : undefined }} />
                  <text x={w >= 0 ? zero + Math.abs(w) + 6 : zero - Math.abs(w) - 6} textAnchor={w >= 0 ? 'start' : 'end'} y={y + 30}
                    style={{ opacity: seen || !anim ? 1 : 0, transition: 'opacity 300ms 700ms' }}>{signedPct(v)}</text>
                </g>
              )
            })}
          </g>
        )
      })}
    </svg>
  )
}

function SizeQuality() {
  const { results } = useData()
  const [hover, setHover] = useState<Model | null>(null)
  const box = useRef<HTMLDivElement>(null)
  const W = 560, H = 330, L = 46, R = 16, T = 16, B = 40
  const x = scaleLinear().domain([4, 16]).range([L, W - R])
  const y = scaleLinear().domain([0.47, 0.57]).range([H - B, T])
  const LABELS: Record<string, string> = { original: 'Original', naive_8bit: '8 bit', per_row: 'Plain 4 bit', g128_l0131: 'Closest', per_row_vkod: 'Best exam', qlora: 'QLoRA' }
  return (
    <div ref={box} style={{ position: 'relative' }}>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Exam score against model size for every model. The 8 GB graphics card limit is marked.">
        {[4, 8, 12, 16].map((t) => (
          <g key={t}><line className="gridline" x1={x(t)} x2={x(t)} y1={T} y2={H - B} /><text x={x(t)} y={H - B + 18} textAnchor="middle">{t} GB</text></g>
        ))}
        {[0.48, 0.5, 0.52, 0.54, 0.56].map((t) => (
          <g key={t}><line className="gridline" x1={L} x2={W - R} y1={y(t)} y2={y(t)} /><text x={L - 8} y={y(t) + 4} textAnchor="end">{pct(t, 0)}</text></g>
        ))}
        <rect x={L} y={T} width={x(8) - L} height={H - B - T} fill="var(--teal)" opacity={0.06} />
        <line x1={x(8)} x2={x(8)} y1={T} y2={H - B} stroke="var(--amber)" strokeWidth={2} />
        <text x={x(8) + 6} y={T + 12} className="strong">8 GB graphics card</text>
        <text x={(L + x(8)) / 2} y={H - 6} textAnchor="middle">fits</text>
        <text x={(x(8) + W - R) / 2} y={H - 6} textAnchor="middle">too large</text>
        {results.models.map((m) => (
          <g key={m.id} tabIndex={0} role="img" aria-label={`${m.label}: ${m.metrics.sizeGb.toFixed(1)} GB, exam ${pct(m.metrics.exam)}`}
            onMouseEnter={() => setHover(m)} onMouseLeave={() => setHover(null)} onFocus={() => setHover(m)} onBlur={() => setHover(null)} style={{ outline: 'none' }}>
            <circle cx={x(m.metrics.sizeGb)} cy={y(m.metrics.exam)} r={hover?.id === m.id ? 9 : 7} fill={FAMILY_COLOR[m.family]}
              stroke="var(--paper)" strokeWidth={2} style={{ transition: 'r 150ms' }} />
            {LABELS[m.id] && <text x={x(m.metrics.sizeGb) + 11} y={y(m.metrics.exam) + 4} className="strong">{LABELS[m.id]}</text>}
          </g>
        ))}
      </svg>
      {hover && box.current && (
        <div className="tooltip" style={{
          left: `${(x(hover.metrics.sizeGb) / W) * 100}%`, top: `${(y(hover.metrics.exam) / H) * 100}%`,
        }}>
          <strong>{hover.label}</strong><br />
          {hover.metrics.sizeGb.toFixed(2)} GB{hover.metrics.sizeMeasured ? '' : ' (estimated)'}, exam {pct(hover.metrics.exam)}
          {hover.metrics.kl !== null && <><br />Distance from the original {hover.metrics.kl.toFixed(3)}</>}
        </div>
      )}
    </div>
  )
}

export default function Results() {
  const { results } = useData()
  return (
    <section id="results" className="section wrap" aria-labelledby="results-title">
      <div className="section-head">
        <h2 id="results-title">How every model scored</h2>
        <p className="lede">
          Distance from the original measures how far a model's predictions are from the original's, word by word, over thousands of predictions:
          0 would be identical. It is the most sensitive measure here. Select a heading to sort.
        </p>
      </div>
      <Leaderboard />
      <p className="note" style={{ marginTop: '0.75rem' }}>
        Darker cells are better within their column. * Estimated size if stored packed; other sizes are measured. {'†'} Lower than the original because QLoRA learned the
        style of its training text, while moving further from the original. Exam: {results.facts.examQuestions} questions from {results.facts.subjects.length} computer science subjects.
      </p>
      <div className="chart-row">
        <div className="panel">
          <h3>What each technique wins back</h3>
          <p className="note">Share of the plain 4 bit model's loss recovered. Teal recovers, red makes it worse.</p>
          <RecoveryChart />
        </div>
        <div className="panel">
          <h3>Score against size</h3>
          <p className="note">Hover or focus a point for details. Everything left of the amber line fits one 8 GB card.</p>
          <SizeQuality />
        </div>
      </div>
    </section>
  )
}
