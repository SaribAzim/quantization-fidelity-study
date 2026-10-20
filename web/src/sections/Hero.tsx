import { useEffect, useMemo, useRef, useState } from 'react'
import { motion } from 'motion/react'
import WeightGrid from '../components/WeightGrid'
import { modelById, useData } from '../data'
import { prefersReducedMotion, useTween } from '../lib'

const GPU_GB = 8
const SCALE_GB = 16

export default function Hero() {
  const { results, weights } = useData()
  const original = modelById(results, 'original')
  const eight = modelById(results, 'naive_8bit')
  const best = modelById(results, 'g128_l0131')

  const stages = useMemo(() => [
    { name: '16 bit', title: 'The original', size: original.metrics.sizeGb, values: weights.grid.original, wdth: 125 },
    { name: '8 bit', title: '8 bit', size: eight.metrics.sizeGb, values: weights.grid.bits8, wdth: 96 },
    { name: '4 bit', title: 'The best 4 bit model', size: best.metrics.sizeGb, values: weights.grid.bits4Groups, wdth: 66 },
  ], [original, eight, best, weights])

  const [stage, setStage] = useState(0)
  const touched = useRef(false)

  // The page's one orchestrated moment: on load, the model is quantized in front of you.
  useEffect(() => {
    if (prefersReducedMotion()) { setStage(2); return }
    const a = window.setTimeout(() => { if (!touched.current) setStage(1) }, 1200)
    const b = window.setTimeout(() => { if (!touched.current) setStage(2) }, 3000)
    return () => { window.clearTimeout(a); window.clearTimeout(b) }
  }, [])

  const choose = (i: number) => { touched.current = true; setStage(i) }
  const current = stages[stage]
  const size = useTween(current.size, 900)
  const fits = current.size <= GPU_GB

  // Distinct values in the first row of the slice, per stage: what "fewer bits" means.
  const distinct = useMemo(() => stages.map((s) => new Set(s.values.slice(0, weights.cols)).size), [stages, weights.cols])

  return (
    <section className="hero wrap" aria-labelledby="hero-title">
      <motion.h1
        id="hero-title"
        className="hero-title"
        initial={{ '--wdth': 125 } as never}
        animate={{ '--wdth': current.wdth } as never}
        transition={{ duration: prefersReducedMotion() ? 0 : 1.1, ease: [0.65, 0, 0.35, 1] }}
      >
        Llama 3.1 8B
      </motion.h1>

      <div className="hero-grid">
        <div>
          <p className="hero-sub">
            In its original 16 bit form this language model needs {original.metrics.sizeGb.toFixed(1)} GB, nearly twice what one 8 GB graphics card holds.
            Rounding every weight to 4 bit shrinks it to {best.metrics.sizeGb.toFixed(1)} GB. This study measures what the rounding costs, and which fixes win the quality back.
          </p>

          <div className="sizebox" aria-live="polite">
            <div className="size-readout">
              <span className="size-number">{size.toFixed(1)} GB</span>
              <span className="size-label">{current.title}</span>
            </div>
            <div className="sizebar" aria-hidden="true">
              <div className="sizebar-fill" style={{ width: `${(size / SCALE_GB) * 100}%`, background: fits ? 'var(--teal)' : 'var(--ember)' }} />
              <div className="sizebar-limit" style={{ left: `${(GPU_GB / SCALE_GB) * 100}%` }} />
            </div>
            <div className="sizebar-scale" aria-hidden="true"><span>0 GB</span><span>8 GB graphics card</span><span>16 GB</span></div>
            <p className="fits">{fits ? <><strong>Fits</strong> on one 8 GB graphics card.</> : <><strong>Does not fit</strong> on one 8 GB graphics card.</>}</p>
          </div>

          <div className="stage-row">
            <div className="segmented" role="group" aria-label="Precision">
              {stages.map((s, i) => (
                <button key={s.name} type="button" aria-pressed={stage === i} onClick={() => choose(i)}>{s.name}</button>
              ))}
            </div>
          </div>

          <div className="hero-actions">
            <a className="btn" href="#answers">Watch the models answer</a>
            <a className="btn quiet" href="#results">See the results</a>
          </div>
        </div>

        <figure className="grid-figure">
          <WeightGrid
            rows={weights.rows}
            cols={weights.cols}
            stages={stages.map((s) => s.values)}
            stage={stage}
            label={`A grid of ${weights.rows} by ${weights.cols} real weights from layer 1 of the model, shown at ${current.name}.`}
          />
          <div className="distinct" aria-label="Distinct values in the first row">
            {stages.map((s, i) => (
              <span key={s.name} data-on={stage === i}>{s.name}: {distinct[i]} values</span>
            ))}
          </div>
          <figcaption className="note grid-caption">
            A real slice of the model: {weights.rows} rows of {weights.cols} weights from layer 1, the part most sensitive to rounding.
            Blue weights are negative, teal ones positive. With fewer bits, each row can use fewer distinct values, so the colours band together.
          </figcaption>
        </figure>
      </div>
    </section>
  )
}
