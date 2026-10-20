// Types for the three data files in public/data, all generated from the research results
// (deployment/export_site_data.py, deployment/build_gallery.py, deployment/export_hero_weights.py).
import { createContext, useContext } from 'react'

export type Metrics = {
  perplexity: number
  kl: number | null
  top1: number
  exam50: number
  exam: number
  examStderr: number
  perSubject: Record<string, number>
  sizeGb: number
  sizeMeasured: boolean
  peakVramGb: number | null
  tokensPerSecond: number | null
}

export type Model = {
  id: string
  label: string
  family: string
  runnable: boolean
  gallery: boolean
  featured: string | null
  note: string
  metrics: Metrics
  recovery: { kl: number | null; exam: number } | null
}

export type Results = {
  generated: string
  facts: { model: string; gpu: string; examQuestions: number; examNoise: number; subjects: string[]; libraryPassages: number }
  baselineId: string
  models: Model[]
  sensitivity: Record<'perRow' | 'groups128', { perLayer: number[]; perKind: Record<string, number> }>
  rag: {
    none: { exam: number; stderr: number }
    plain: { exam: number; stderr: number; fixed: number; broken: number; p: number }
    hyde: { exam: number; stderr: number; fixed: number; broken: number; p: number }
  }
  qlora: { train: [number, number][]; validation: [number, number][] }
  gptqWindow: { label: string; short: number; long: number }[]
}

export type Answer = {
  text: string
  pieces: string[]
  times: number[]
  tokensPerSecond: number | null
  identicalTokens: number
  wordSimilarity: number
  diff: [string, boolean][]
}

export type GalleryPrompt = { id: string; topic: string; prompt: string; answers: Record<string, Answer> }
export type Gallery = { prompts: GalleryPrompt[]; models: Record<string, { loadSeconds: number | null }> }

export type Weights = {
  rows: number
  cols: number
  grid: { original: number[]; bits8: number[]; bits4: number[]; bits4Groups: number[] }
  explain: {
    row: number
    original: number[]
    perRow: number[]
    groups: number[]
    bits8: number[]
    levelsPerRow: number[]
    levelsGroup: number[]
    levels8: number[]
    rowMin: number
    rowMax: number
    groupMin: number
    groupMax: number
    meanErrorPerRow: number
    meanErrorGroups: number
    meanError8bit: number
  }
}

export type SiteData = { results: Results; gallery: Gallery; weights: Weights }

const base = import.meta.env.BASE_URL

export async function loadSiteData(): Promise<SiteData> {
  const get = (name: string) => fetch(`${base}data/${name}.json`).then((r) => {
    if (!r.ok) throw new Error(`Could not load ${name}.json (${r.status})`)
    return r.json()
  })
  const [results, gallery, weights] = await Promise.all([get('results'), get('gallery'), get('weights')])
  return { results, gallery, weights }
}

export const DataContext = createContext<SiteData | null>(null)

export function useData(): SiteData {
  const data = useContext(DataContext)
  if (!data) throw new Error('useData must be used inside the data provider')
  return data
}

export function modelById(results: Results, id: string): Model {
  const m = results.models.find((x) => x.id === id)
  if (!m) throw new Error(`Unknown model ${id}`)
  return m
}

// The two best 4 bit models, which visitors can test, and the four models of the showcase.
export const BEST_TWO = ['g128_l0131', 'per_row_vkod'] as const
export const SHOWCASE = ['original', 'naive_8bit', 'g128_l0131', 'per_row_vkod'] as const

export const REPO = 'https://github.com/SaribAzim/quantization-fidelity-study'
export const code = (path: string) => `${REPO}/blob/main/${path}`

// Colours per family of model, used by every chart so a family keeps its colour everywhere.
export const FAMILY_COLOR: Record<string, string> = {
  Original: 'var(--blue)',
  '8 bit': 'var(--slate)',
  'Naive, one ruler per row': 'var(--ember)',
  'Naive, groups of 128': 'var(--teal-soft)',
  'Mixed precision': 'var(--teal)',
  GPTQ: 'var(--violet)',
  QLoRA: 'var(--berry)',
}
