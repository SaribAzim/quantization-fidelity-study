import { useEffect, useMemo, useRef } from 'react'
import { cssVar, useColorScheme } from '../lib'

type Props = {
  rows: number
  cols: number
  stages: number[][]
  /** 0 = first stage, 1 = second, 2 = third; fractions blend between them with a left to right sweep. */
  position: number
  /** 0 to 1: the grid develops in a diagonal wave when the page opens. */
  intro: number
  label: string
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '')
  const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

const smooth = (p: number) => p * p * (3 - 2 * p)
const clamp01 = (v: number) => Math.max(0, Math.min(1, v))

/**
 * Draws real weights as tiles: ink blue for negative, mustard for positive, paler near zero.
 * Between two stages each column switches a little after the one to its left,
 * so a scroll sweeps the rounding across the grid and each row visibly loses colours.
 */
export default function WeightGrid({ rows, cols, stages, position, intro, label }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const scheme = useColorScheme()

  // Clip the colour scale at the 98th percentile so a few extreme weights do not wash out the rest.
  const clip = useMemo(() => {
    const mags = stages[0].map(Math.abs).sort((a, b) => a - b)
    return mags[Math.floor(mags.length * 0.98)] || 1
  }, [stages])

  const draw = () => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap) return
    const width = wrap.clientWidth
    const tileW = width / cols
    const tileH = Math.max(4, tileW * 1.25)
    const height = Math.round(tileH * rows)
    const dpr = window.devicePixelRatio || 1
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
      canvas.style.height = `${height}px`
    }
    const ctx = canvas.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const surface = hexToRgb(cssVar('--surface') || '#fbf7ef')
    const neg = hexToRgb(cssVar('--blue') || '#2c4a7a')
    const pos = hexToRgb(cssVar('--amber') || '#c9962b')
    ctx.fillStyle = `rgb(${surface.join(',')})`
    ctx.fillRect(0, 0, width, height)
    const gap = tileW > 4 ? 1 : 0
    const from = Math.min(stages.length - 2, Math.floor(position))
    const f = position - from
    const a = stages[from], b = stages[from + 1]
    for (let c = 0; c < cols; c++) {
      const sweep = smooth(clamp01(f * 1.6 - (c / cols) * 0.6))
      for (let r = 0; r < rows; r++) {
        const i = r * cols + c
        const v = a[i] + (b[i] - a[i]) * sweep
        const t = Math.max(-1, Math.min(1, v / clip))
        const develop = smooth(clamp01(intro * 2.2 - ((r / rows) + (c / cols)) * 0.6))
        const m = Math.pow(Math.abs(t), 0.7) * develop
        const target = t < 0 ? neg : pos
        const rgb = surface.map((p, k) => Math.round(p + (target[k] - p) * m))
        ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`
        ctx.fillRect(c * tileW, r * tileH, tileW - gap, tileH - gap)
      }
    }
  }

  const drawRef = useRef(draw)
  drawRef.current = draw
  useEffect(() => { draw() })

  // Redraw on resize and when the colour scheme changes.
  useEffect(() => {
    const ro = new ResizeObserver(() => drawRef.current())
    if (wrapRef.current) ro.observe(wrapRef.current)
    return () => ro.disconnect()
  }, [scheme])

  return (
    <div ref={wrapRef}>
      <canvas ref={canvasRef} className="grid-canvas" role="img" aria-label={label} />
    </div>
  )
}
