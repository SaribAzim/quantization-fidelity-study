import { ANSWERS_PAGE, BEST, modelById, useData } from '../data'
import { pct } from '../lib'

/** The closing band of the project page: an invitation to the model answers page. */
export default function NextPage() {
  const { results } = useData()
  const best = modelById(results, BEST)
  return (
    <section className="next-band" aria-labelledby="next-title">
      <div className="wrap next-inner">
        <div>
          <h2 id="next-title">See the difference for yourself</h2>
          <p className="lede">
            Watch the original, the 8 bit model and the best 4 bit model answer the same questions, word by word. The 4 bit model is
            {' '}{best.metrics.sizeGb.toFixed(1)} GB, scores {pct(best.metrics.exam)} on the exam, and picks the same next word as the original {pct(best.metrics.top1)} of the time.
          </p>
        </div>
        <a className="btn big" href={ANSWERS_PAGE}>Watch the models answer</a>
      </div>
    </section>
  )
}
