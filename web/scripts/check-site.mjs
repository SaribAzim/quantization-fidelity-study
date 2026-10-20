// Site check: start `npm run preview` first, then `npm run check`.
// 1. Fails if any visible text (outside quoted model answers, marked data-quote) contains a hyphen or dash.
// 2. Fails on browser console errors.
// 3. Saves screenshots for design review in screenshots/.
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'

const URL = process.env.SITE_URL || 'http://localhost:4173/quantization-fidelity-study/'
const DASHES = /[-‐‑‒–—―]/g
const VIEWS = [
  { name: 'desktop', width: 1440, height: 900, colorScheme: 'light' },
  { name: 'tablet', width: 820, height: 1180, colorScheme: 'light' },
  { name: 'phone', width: 390, height: 844, colorScheme: 'light' },
  { name: 'desktop_dark', width: 1440, height: 900, colorScheme: 'dark' },
]
mkdirSync('screenshots', { recursive: true })

const browser = await chromium.launch()
let failures = 0

async function scrollThrough(page) {
  const height = await page.evaluate(() => document.body.scrollHeight)
  for (let y = 0; y < height; y += 500) {
    await page.evaluate((top) => window.scrollTo(0, top), y)
    await page.waitForTimeout(60)
  }
  await page.evaluate(() => window.scrollTo(0, 0))
}

for (const view of VIEWS) {
  const context = await browser.newContext({ viewport: { width: view.width, height: view.height }, colorScheme: view.colorScheme, reducedMotion: 'reduce' })
  const page = await context.newPage()
  const errors = []
  // On localhost the page looks for the live backend; a refused connection there just means it is not running.
  page.on('console', (m) => { if (m.type() === 'error' && !m.location().url.startsWith('http://127.0.0.1:8000')) errors.push(m.text()) })
  page.on('pageerror', (e) => errors.push(String(e)))
  await page.goto(URL, { waitUntil: 'networkidle' })
  await page.waitForSelector('#hero-title')
  await scrollThrough(page)

  if (view.name === 'desktop') {
    const text = await page.evaluate(() => {
      const clone = document.body.cloneNode(true)
      clone.querySelectorAll('[data-quote], script, style, noscript').forEach((n) => n.remove())
      document.body.appendChild(Object.assign(document.createElement('div'), { id: '__probe', hidden: true }))
      return clone.innerText ?? clone.textContent
    })
    const hits = [...text.matchAll(DASHES)].map((m) => text.slice(Math.max(0, m.index - 40), m.index + 40).replace(/\s+/g, ' '))
    if (hits.length) {
      failures++
      console.log(`FAIL: ${hits.length} hyphen or dash characters in visible text:`)
      hits.slice(0, 20).forEach((h) => console.log(`  ...${h}...`))
    } else {
      console.log('OK: no hyphens or dashes in visible text (quoted model answers excluded)')
    }
  }

  await page.screenshot({ path: `screenshots/${view.name}.png`, fullPage: true })
  // Fresh hero and the showcase after a replay, for closer review.
  await page.screenshot({ path: `screenshots/${view.name}_hero.png` })
  const play = page.getByRole('button', { name: 'Play the answers' }).first()
  await play.click()
  await page.waitForTimeout(400)
  await page.locator('#answers').screenshot({ path: `screenshots/${view.name}_answers.png` })

  if (errors.length) { failures++; console.log(`FAIL (${view.name}): browser errors:`, errors.slice(0, 5)) }
  else console.log(`OK (${view.name}): no browser errors, screenshots saved`)
  await context.close()
}

// The hero animation itself, without reduced motion: start, middle and end of the load sequence.
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } })
const page = await context.newPage()
await page.goto(URL, { waitUntil: 'networkidle' })
for (const [ms, name] of [[300, 'start'], [2000, 'middle'], [4600, 'end']]) {
  await page.waitForTimeout(ms === 300 ? 300 : ms === 2000 ? 1700 : 2600)
  await page.screenshot({ path: `screenshots/hero_motion_${name}.png` })
}
await context.close()
await browser.close()
console.log(failures ? `\n${failures} check(s) failed.` : '\nAll site checks passed.')
process.exit(failures ? 1 : 0)
