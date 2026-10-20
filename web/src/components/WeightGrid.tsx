import { useEffect, useMemo, useRef } from 'react'
import { cssVar, prefersReducedMotion, useColorScheme } from '../lib'

type Props = { rows: number; cols: number; stages: number[][]; stage: number; label: string }

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '')
  const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

/**
 * Draws real weights as tiles: blue for negative, teal for positive, paler near zero.
 * When the stage changes, tiles sweep from left to right into their rounded values,
 * so you can watch each row lose distinct colours.
 */
export default function WeightGrid({ rows, cols, stages, stage, label }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const scheme = useColorScheme()
  const shown = useRef<number[]>(stages[stage].slice())
  const frame = useRef(0)

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
    const tileH = Math.max(4, tileW * 1.7)
    const height = Math.round(tileH * rows)
    const dpr = window.devicePixelRatio || 1
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
      canvas.style.height = `${height}px`
    }
    const ctx = canvas.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const paper = hexToRgb(cssVar('--paper') || '#ffffff')
    const neg = hexToRgb(cssVar('--blue') || '#1f57d6')
    const pos = hexToRgb(cssVar('--teal') || '#0f8c7e')
    ctx.fillStyle = `rgb(${paper.join(',')})`
    ctx.fillRect(0, 0, width, height)
    const gap = tileW > 4 ? 1 : 0
    const values = shown.current
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = values[r * cols + c]
        const t = Math.max(-1, Math.min(1, v / clip))
        const m = Math.pow(Math.abs(t), 0.7)
        const target = t < 0 ? neg : pos
        const rgb = paper.map((p, i) => Math.round(p + (target[i] - p) * m))
        ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`
        ctx.fillRect(c * tileW, r * tileH, tileW - gap, tileH - gap)
      }
    }
  }

  // Animate between stages: a left to right sweep, about one second.
  useEffect(() => {
    const from = shown.current.slice()
    const to = stages[stage]
    cancelAnimationFrame(frame.current)
    if (prefersReducedMotion()) {
      shown.current = to.slice()
      draw()
      return
    }
    const start = performance.now()
    const total = 1100
    const tick = (now: number) => {
      const elapsed = now - start
      const next = new Array(to.length)
      for (let c = 0; c < cols; c++) {
        const p = Math.max(0, Math.min(1, (elapsed - (c / cols) * total * 0.5) / (total * 0.5)))
        const e = p * p * (3 - 2 * p)
        for (let r = 0; r < rows; r++) {
          const i = r * cols + c
          next[i] = from[i] + (to[i] - from[i]) * e
        }
      }
      shown.current = next
      draw()
      if (elapsed < total) frame.current = requestAnimationFrame(tick)
    }
    frame.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage, stages])

  // Redraw on resize and when the colour scheme changes.
  useEffect(() => {
    draw()
    const ro = new ResizeObserver(() => draw())
    if (wrapRef.current) ro.observe(wrapRef.current)
    return () => ro.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scheme])

  return (
    <div ref={wrapRef}>
      <canvas ref={canvasRef} className="grid-canvas" role="img" aria-label={label} />
    </div>
  )
}
