import { useEffect, useRef, useState } from 'react'
import { scaleLinear } from 'd3-scale'
import { useData } from '../data'
import { num, useTween } from '../lib'

const lerp = (a: number, b: number, t: number) => a + (b - a) * t

function RulerFigure({ step }: { step: number }) {
  const { weights } = useData()
  const e = weights.explain
  const pad = (lo: number, hi: number): [number, number] => { const d = (hi - lo) * 0.06; return [lo - d, hi + d] }
  const domains = [pad(e.rowMin, e.rowMax), pad(e.groupMin, e.groupMax), pad(e.groupMin, e.groupMax)]
  const rounded = [e.perRow, e.groups, e.bits8]
  const levels = [e.levelsPerRow, e.levelsGroup, e.levels8]
  const errors = [e.meanErrorPerRow, e.meanErrorGroups, e.meanError8bit]
  const ranges = [[e.rowMin, e.rowMax], [e.groupMin, e.groupMax], [e.groupMin, e.groupMax]]

  // A single number tweens between steps; everything on the figure is interpolated from it.
  const sf = useTween(step, 900)
  const i0 = Math.floor(sf), i1 = Math.min(2, i0 + 1), f = sf - i0
  const lo = lerp(domains[i0][0], domains[i1][0], f), hi = lerp(domains[i0][1], domains[i1][1], f)

  const W = 640, H = 252, L = 24, R = 24, yTop = 44, yRuler = 192, laneGap = 9
  const x = scaleLinear().domain([lo, hi]).range([L, W - R])
  const visibleLevels = levels[step].filter((v) => v >= lo && v <= hi)

  // Beeswarm: weights that would overlap move down a lane, so all 40 stay visible.
  const lanes: number[] = []
  const laneOf = new Array<number>(e.original.length)
  e.original.map((v, k) => [x(v), k] as const).sort((a, b) => a[0] - b[0]).forEach(([px, k]) => {
    let lane = lanes.findIndex((last) => px - last >= laneGap)
    if (lane < 0 || lane > 7) lane = lanes.length > 7 ? lanes.indexOf(Math.min(...lanes)) : lanes.length
    lanes[lane] = px
    laneOf[k] = lane
  })
  // How many weights land on each mark once rounded.
  const landed = new Map<number, number>()
  rounded[step].forEach((v) => landed.set(v, (landed.get(v) ?? 0) + 1))
  const extremes = [e.rowMin, e.rowMax].filter((v) => v >= lo && v <= hi)

  return (
    <div className="sticky-figure">
      <svg className="ruler" viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label={`Forty real weights rounded to the nearest allowed value. Average rounding error ${num(errors[step], 4)}.`}>
        <text x={L} y={22}>Forty real weights from one row of layer 1</text>
        <text x={L} y={yRuler + 44}>Allowed values, with how many weights land on each</text>
        <line x1={L} x2={W - R} y1={yRuler} y2={yRuler} stroke="var(--muted)" />
        {visibleLevels.map((v, k) => (
          <line key={`${step}-${k}`} x1={x(v)} x2={x(v)} y1={yRuler - (step === 2 ? 7 : 13)} y2={yRuler + (step === 2 ? 7 : 13)}
            stroke="var(--ink)" strokeWidth={step === 2 ? 1 : 1.6} opacity={step === 2 ? 0.55 : 0.9} />
        ))}
        {e.original.map((v, k) => {
          const r = lerp(rounded[i0][k], rounded[i1][k], f)
          const cy = yTop + laneOf[k] * laneGap
          return (
            <g key={k}>
              <line x1={x(v)} y1={cy + 4} x2={x(r)} y2={yRuler - 16} stroke="var(--ember)" strokeOpacity={0.35} strokeWidth={1} />
              <circle cx={x(v)} cy={cy} r={3.6} fill="var(--ink)" />
              <circle cx={x(r)} cy={yRuler} r={4.2} fill="var(--sage)" />
            </g>
          )
        })}
        {f === 0 && step < 2 && [...landed].filter(([, n]) => n > 1).map(([v, n]) => (
          <text key={v} className="landed" x={x(v)} y={yRuler + 26} textAnchor="middle">{n}</text>
        ))}
        {f === 0 && step === 0 && extremes.map((v) => (
          <g key={v}>
            <circle cx={x(v)} cy={yTop} r={5} fill="none" stroke="var(--ember)" strokeWidth={1.6} />
            <text x={x(v)} y={yTop + 22} textAnchor={v > 0 ? 'end' : 'start'}>{v > 0 ? 'Largest weight in the row' : 'Smallest'}</text>
          </g>
        ))}
        <text x={L} y={H - 4}>{num(lo, 3)}</text>
        <text x={W - R} y={H - 4} textAnchor="end">{num(hi, 3)}</text>
      </svg>
      <div className="ruler-stats">
        <div><div className="stat-value">{step === 2 ? 256 : 16}</div><div className="stat-label">allowed values per ruler</div></div>
        <div><div className="stat-value">{num(ranges[step][0], 3)} to {num(ranges[step][1], 3)}</div><div className="stat-label">range the ruler covers</div></div>
        <div><div className="stat-value">{num(errors[step], 4)}</div><div className="stat-label">average rounding error</div></div>
      </div>
    </div>
  )
}

export default function Rounding() {
  const { weights } = useData()
  const e = weights.explain
  const [active, setActive] = useState(0)
  const stepRefs = useRef<(HTMLLIElement | null)[]>([])

  // The visitor's scroll chooses the step: whichever step text crosses the middle of the screen.
  useEffect(() => {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) setActive(Number((entry.target as HTMLElement).dataset.step))
      })
    }, { rootMargin: '-45% 0px -45% 0px' })
    stepRefs.current.forEach((el) => el && io.observe(el))
    return () => io.disconnect()
  }, [])

  const shrink = (e.meanErrorPerRow / e.meanErrorGroups).toFixed(1)
  const steps = [
    { title: 'One ruler per row', text: `At 4 bit, each row of weights may use only 16 values, spread evenly between its smallest and largest weight. In this row a single extreme weight stretches the ruler from ${num(e.rowMin, 3)} to ${num(e.rowMax, 3)}, so nearly all the ordinary weights fall between two marks and are rounded hard.` },
    { title: 'A ruler for every 128 weights', text: `Giving every group of 128 neighbouring weights its own ruler keeps the 16 marks close to the weights they serve. On this row the average rounding error shrinks ${shrink} times, for about 0.4 GB more storage across the model.` },
    { title: 'Keep the fragile parts at 8 bit', text: `8 bit allows 256 values per ruler, and the error almost disappears. Doing this only where rounding hurts most, the v, k, o and down tables, or layers 0, 1 and 31, costs well under 1 GB and recovers most of the lost quality.` },
  ]

  return (
    <section className="section wrap" aria-labelledby="rounding-title">
      <div className="section-head">
        <h2 id="rounding-title">How a weight gets rounded</h2>
        <p className="lede">Quantizing means storing each weight as the nearest mark on a coarse ruler. Scroll to see why the size of the ruler matters.</p>
      </div>
      <div className="scrolly">
        <RulerFigure step={active} />
        <ol className="steps">
          {steps.map((s, i) => (
            <li key={s.title} className="step" data-step={i} data-active={active === i} ref={(el) => { stepRefs.current[i] = el }}>
              <h3><span className="step-num">{i + 1}</span>{s.title}</h3>
              <p>{s.text}</p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  )
}
