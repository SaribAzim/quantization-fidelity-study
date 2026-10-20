import { useEffect, useRef, useState } from 'react'

// Numbers are written with a true minus sign (U+2212), never a hyphen: the site's copy rule.
export const MINUS = '−'
export function num(value: number, digits = 2): string {
  const s = Math.abs(value).toFixed(digits)
  return value < 0 ? MINUS + s : s
}
export const pct = (value: number, digits = 1) => `${num(value * 100, digits)}%`
export const signedPct = (value: number, digits = 0) => `${value >= 0 ? '+' : MINUS}${Math.abs(value * 100).toFixed(digits)}%`
export const gb = (value: number) => `${value.toFixed(1)} GB`

export function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

/** True once the element has scrolled into view (and stays true). Used to draw each chart once. */
export function useSeenOnce<T extends Element>(threshold = 0.25): [React.RefObject<T | null>, boolean] {
  const ref = useRef<T>(null)
  const [seen, setSeen] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el || seen) return
    const io = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) { setSeen(true); io.disconnect() }
    }, { threshold })
    io.observe(el)
    return () => io.disconnect()
  }, [seen, threshold])
  return [ref, seen]
}

const easeInOut = (t: number) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2)

/** Smoothly animates a number toward `target` (instantly when reduced motion is on). */
export function useTween(target: number, duration = 700): number {
  const [value, setValue] = useState(target)
  const from = useRef(target)
  useEffect(() => {
    if (prefersReducedMotion() || duration <= 0) { setValue(target); from.current = target; return }
    const start = performance.now()
    const begin = from.current
    let frame = 0
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration)
      const v = begin + (target - begin) * easeInOut(t)
      from.current = v
      setValue(v)
      if (t < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [target, duration])
  return value
}

/** Reads a CSS custom property, so canvas drawing follows light and dark mode. */
export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

export function useColorScheme(): 'light' | 'dark' {
  const query = '(prefers-color-scheme: dark)'
  const [scheme, setScheme] = useState<'light' | 'dark'>(() => (window.matchMedia(query).matches ? 'dark' : 'light'))
  useEffect(() => {
    const mq = window.matchMedia(query)
    const on = () => setScheme(mq.matches ? 'dark' : 'light')
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  return scheme
}
