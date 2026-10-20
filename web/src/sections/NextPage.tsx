import { BEST, modelById, TEST_PAGE, useData } from '../data'
import { pct } from '../lib'

/** The closing band of the project page: an invitation to the test page. */
export default function NextPage() {
  const { results } = useData()
  const best = modelById(results, BEST)
  return (
    <section className="next-band" aria-labelledby="next-title">
      <div className="wrap next-inner">
        <div>
          <h2 id="next-title">See the difference for yourself</h2>
          <p className="lede">
            Watch the original, the 8 bit model and the best 4 bit model answer the same question, then test the 4 bit model:
            {' '}{best.metrics.sizeGb.toFixed(1)} GB, {pct(best.metrics.exam)} on the exam, and the same next word as the original {pct(best.metrics.top1)} of the time.
          </p>
        </div>
        <a className="btn big" href={TEST_PAGE}>Test the model</a>
      </div>
    </section>
  )
}
