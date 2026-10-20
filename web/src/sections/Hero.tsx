import { useEffect, useMemo, useRef, useState } from 'react'
import { motion } from 'motion/react'
import WeightGrid from '../components/WeightGrid'
import { ANSWERS_PAGE, modelById, useData } from '../data'
import { prefersReducedMotion } from '../lib'

const GPU_GB = 8
const SCALE_GB = 16
const TITLE = 'Llama 3.1 8B'

const clamp01 = (v: number) => Math.max(0, Math.min(1, v))
const smooth = (p: number) => p * p * (3 - 2 * p)
const lerp3 = (v: [number, number, number], pos: number) => {
  const i = Math.min(1, Math.floor(pos))
  return v[i] + (v[i + 1] - v[i]) * (pos - i)
}

// Scroll progress through the pinned hero, mapped to precision: hold at 16 bit,
// quantize to 8 bit, hold, quantize to 4 bit, hold. The holds let each state be read.
const KEYS: [number, number][] = [[0.1, 0], [0.4, 1], [0.55, 1], [0.85, 2]]
function positionFrom(p: number): number {
  if (p <= KEYS[0][0]) return 0
  for (let k = 0; k < KEYS.length - 1; k++) {
    const [p0, s0] = KEYS[k], [p1, s1] = KEYS[k + 1]
    if (p <= p1) return s0 + (s1 - s0) * smooth(clamp01((p - p0) / (p1 - p0)))
  }
  return 2
}
const STOPS = [0, 0.48, 0.93] // scroll progress that shows each stage fully

export default function Hero() {
  const { results, weights } = useData()
  const original = modelById(results, 'original')
  const eight = modelById(results, 'naive_8bit')
  const best = modelById(results, 'g128_l0131')
  const reduced = useMemo(() => prefersReducedMotion(), [])

  const sectionRef = useRef<HTMLElement>(null)
  const [progress, setProgress] = useState(reduced ? 1 : 0)
  const [picked, setPicked] = useState(2) // used only when motion is reduced
  const [intro, setIntro] = useState(reduced ? 1 : 0)

  // The opening moment: the grid develops like a photograph while the title rises in.
  useEffect(() => {
    if (reduced) return
    let frame = 0
    const start = performance.now()
    const tick = (now: number) => {
      const t = clamp01((now - start - 250) / 1700)
      setIntro(t)
      if (t < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [reduced])

  // The visitor's scroll quantizes the model.
  useEffect(() => {
    if (reduced) return
    let frame = 0
    const measure = () => {
      frame = 0
      const el = sectionRef.current
      if (!el) return
      const rect = el.getBoundingClientRect()
      const travel = rect.height - window.innerHeight
      setProgress(clamp01(-rect.top / Math.max(1, travel)))
    }
    const onScroll = () => { if (!frame) frame = requestAnimationFrame(measure) }
    measure()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    return () => { window.removeEventListener('scroll', onScroll); window.removeEventListener('resize', onScroll); cancelAnimationFrame(frame) }
  }, [reduced])

  const position = reduced ? picked : positionFrom(progress)
  const stage = Math.round(position)

  const stages = useMemo(() => [
    { name: '16 bit', title: 'The original, 16 bit', values: weights.grid.original },
    { name: '8 bit', title: 'Naive 8 bit', values: weights.grid.bits8 },
    { name: '4 bit', title: 'The best 4 bit model', values: weights.grid.bits4Groups },
  ], [weights])
  // Show a 16 by 64 corner of the slice with larger tiles, so the banding at 4 bit is easy to see.
  // The weights were rounded on full rows, so the corner is exactly what the model stores.
  const VIEW_ROWS = Math.min(16, weights.rows), VIEW_COLS = Math.min(64, weights.cols)
  const view = useMemo(() => stages.map((s) => {
    const out: number[] = []
    for (let r = 0; r < VIEW_ROWS; r++) out.push(...s.values.slice(r * weights.cols, r * weights.cols + VIEW_COLS))
    return out
  }), [stages, weights.cols, VIEW_ROWS, VIEW_COLS])
  const size = lerp3([original.metrics.sizeGb, eight.metrics.sizeGb, best.metrics.sizeGb], position)
  const wdth = lerp3([125, 96, 66], position)
  const fits = size <= GPU_GB
  const distinct = useMemo(() => stages.map((s) => new Set(s.values.slice(0, weights.cols)).size), [stages, weights.cols])

  // Choosing a stage scrolls to it, so the buttons and the scroll never disagree.
  const choose = (i: number) => {
    if (reduced) { setPicked(i); return }
    const el = sectionRef.current
    if (!el) return
    const top = el.getBoundingClientRect().top + window.scrollY
    window.scrollTo({ top: top + STOPS[i] * (el.offsetHeight - window.innerHeight), behavior: 'smooth' })
  }

  return (
    <section ref={sectionRef} className={`hero${reduced ? '' : ' pinned'}`} aria-labelledby="hero-title">
      <div className="hero-pin">
        <div className="wrap hero-inner">
          <h1 id="hero-title" className="hero-title" aria-label={TITLE} style={{ ['--wdth' as never]: wdth }}>
            {[...TITLE].map((ch, i) => (
              <span key={i} className={ch === ' ' ? 'hero-char hero-space' : 'hero-char'} aria-hidden="true">
                <motion.span
                  initial={reduced ? false : { y: '105%' }}
                  animate={{ y: '0%' }}
                  transition={{ duration: 0.9, delay: 0.15 + i * 0.045, ease: [0.22, 1, 0.36, 1] }}
                >{ch === ' ' ? ' ' : ch}</motion.span>
              </span>
            ))}
          </h1>

          <div className="hero-grid">
            <motion.div
              initial={reduced ? false : { opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.9, ease: [0.22, 1, 0.36, 1] }}
            >
              <p className="hero-sub">
                In 16 bit this language model needs {original.metrics.sizeGb.toFixed(1)} GB, nearly twice what one 8 GB graphics card holds.
                This study rounds it down to 4 bit, measures what the rounding costs, and finds which fixes win the quality back.
              </p>

              <div className="sizebox" aria-live="polite">
                <div className="size-readout">
                  <span className="size-number">{size.toFixed(1)} GB</span>
                  <span className="size-label">{stages[stage].title}</span>
                </div>
                <div className="sizebar" aria-hidden="true">
                  <div className="sizebar-fill" style={{ width: `${(size / SCALE_GB) * 100}%`, background: fits ? 'var(--sage)' : 'var(--ember)' }} />
                  <div className="sizebar-limit" style={{ left: `${(GPU_GB / SCALE_GB) * 100}%` }} />
                </div>
                <div className="sizebar-scale" aria-hidden="true"><span>0 GB</span><span>8 GB graphics card</span><span>16 GB</span></div>
                <p className="fits">{fits ? <><strong>Fits</strong> on one 8 GB graphics card.</> : <><strong>Does not fit</strong> on one 8 GB graphics card.</>}</p>
              </div>

              <div className="hero-actions">
                <a className="btn" href={ANSWERS_PAGE}>Watch it answer</a>
                <a className="btn quiet" href="#study">Read the study</a>
              </div>
            </motion.div>

            <motion.figure
              className="grid-figure"
              initial={reduced ? false : { opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.4, delay: 0.2 }}
            >
              <WeightGrid rows={VIEW_ROWS} cols={VIEW_COLS} stages={view} position={position} intro={intro}
                label={`A grid of ${VIEW_ROWS} by ${VIEW_COLS} real weights from layer 1 of the model, shown at ${stages[stage].name}.`} />
              <div className="distinct" aria-label="Distinct values in the first row">
                {stages.map((s, i) => (
                  <span key={s.name} data-on={stage === i}>{s.name}: {distinct[i]} values</span>
                ))}
              </div>
              <figcaption className="note grid-caption">
                Real weights from layer 1 of the model, the part most sensitive to rounding: {VIEW_ROWS} rows of {VIEW_COLS}, each row rounded as the model stores it.
                Blue weights are negative, gold ones positive. With fewer bits each row may use fewer distinct values, so the colours band together.
              </figcaption>
            </motion.figure>
          </div>

          <div className="rail" role="group" aria-label="Precision">
            <div className="rail-track" aria-hidden="true">
              <div className="rail-fill" style={{ width: `${(position / 2) * 100}%` }} />
            </div>
            {stages.map((s, i) => (
              <button key={s.name} type="button" className="rail-stop" style={{ left: `${i * 50}%` }} aria-pressed={stage === i} onClick={() => choose(i)}>
                <span className="rail-dot" aria-hidden="true" />{s.name}
              </button>
            ))}
            {!reduced && <span className="rail-hint" style={{ opacity: progress < 0.04 ? 1 : 0 }}>Scroll to quantize the model</span>}
          </div>
        </div>
      </div>
    </section>
  )
}
