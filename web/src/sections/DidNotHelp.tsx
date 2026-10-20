import type { ReactNode } from 'react'
import { modelById, useData } from '../data'
import { pct, prefersReducedMotion, useSeenOnce } from '../lib'

function FixedBroken() {
  const { results } = useData()
  const [ref, seen] = useSeenOnce<SVGSVGElement>(0.4)
  const rows = [{ name: 'Plain retrieval', r: results.rag.plain }, { name: 'HyDE', r: results.rag.hyde }]
  const W = 520, H = 120, mid = 300, unit = 2.4
  const anim = !prefersReducedMotion()
  return (
    <svg ref={ref} className="chart" viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
      aria-label={rows.map((x) => `${x.name}: ${x.r.fixed} answers fixed, ${x.r.broken} broken`).join('. ')}>
      <text x={mid - 6} y={14} textAnchor="end">answers broken</text>
      <text x={mid + 6} y={14}>answers fixed</text>
      {rows.map((row, i) => {
        const y = 28 + i * 44
        const fw = seen || !anim ? row.r.fixed * unit : 0
        const bw = seen || !anim ? row.r.broken * unit : 0
        return (
          <g key={row.name}>
            <text className="strong" x={0} y={y + 18}>{row.name}</text>
            <rect x={mid - bw} y={y} width={bw} height={26} rx={3} fill="var(--ember)" style={{ transition: anim ? 'all 800ms cubic-bezier(.65,0,.35,1)' : undefined }} />
            <rect x={mid} y={y} width={fw} height={26} rx={3} fill="var(--teal)" style={{ transition: anim ? 'all 800ms cubic-bezier(.65,0,.35,1)' : undefined }} />
            <text x={mid - bw - 6} y={y + 18} textAnchor="end">{row.r.broken}</text>
            <text x={mid + fw + 6} y={y + 18}>{row.r.fixed}</text>
          </g>
        )
      })}
    </svg>
  )
}

/** Highlights the space QLoRA puts before punctuation, the formatting of its WikiText training text. */
function SpacedPunctuation({ text }: { text: string }) {
  const parts: ReactNode[] = []
  const re = /(\w)( )([,.;:])/g
  let last = 0
  let m: RegExpExecArray | null
  let k = 0
  while ((m = re.exec(text))) {
    parts.push(text.slice(last, m.index + 1))
    parts.push(<span key={k++} className="gap"> </span>)
    last = m.index + 2
  }
  parts.push(text.slice(last))
  return <>{parts}</>
}

export default function DidNotHelp() {
  const { results, gallery } = useData()
  const plain = modelById(results, 'per_row').metrics
  const qlora = modelById(results, 'qlora').metrics
  const original = modelById(results, 'original').metrics
  const example = gallery.prompts.find((p) => p.id === 'prime_221') ?? gallery.prompts[0]
  const quote = example.answers.qlora?.text ?? ''
  return (
    <section className="section wrap" aria-labelledby="nothelp-title">
      <div className="section-head">
        <h2 id="nothelp-title">What did not help</h2>
        <p className="lede">Two popular ideas were tested on the plain 4 bit model and did not recover the loss. Both are reported as measured.</p>
      </div>
      <div className="two-col">
        <div className="panel">
          <h3>Looking things up</h3>
          <p>
            The model was given passages from {results.facts.libraryPassages.toLocaleString('en')} Wikipedia computer science articles before answering.
            HyDE first writes a draft answer and searches with it. Retrieval repaired some damaged answers but broke as many correct ones:
            the exam stayed at {pct(results.rag.hyde.exam)} with HyDE and fell to {pct(results.rag.plain.exam)} with plain retrieval.
          </p>
          <FixedBroken />
        </div>
        <div className="panel">
          <h3>Retraining small adapters</h3>
          <p>
            QLoRA trained small extra weights on Wikipedia text. Perplexity fell to {qlora.perplexity.toFixed(2)}, below even the original's {original.perplexity.toFixed(2)},
            yet the distance from the original grew from {plain.kl!.toFixed(3)} to {qlora.kl!.toFixed(3)} and the exam fell to {pct(qlora.exam)}.
            It learned the style of its training text, and you can see it: its answers put a space before commas and full stops, exactly as that text is stored.
          </p>
          <p className="note" style={{ marginBottom: '0.4rem' }}>Its recorded answer to "{example.prompt.split('\n')[0]}", spaces before punctuation marked:</p>
          <div className="quote" data-quote="model answer"><SpacedPunctuation text={quote} /></div>
        </div>
      </div>
    </section>
  )
}
