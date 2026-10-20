// Site check: start `npm run preview` first, then `npm run check`.
// 1. Fails if any visible text (outside quoted model answers, marked data-quote) contains a hyphen or dash.
// 2. Fails on browser console errors.
// 3. Saves screenshots for design review in screenshots/: both pages, four views, and the hero at each scroll stage.
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'

const ROOT = process.env.SITE_URL || 'http://localhost:4173/quantization-fidelity-study/'
const PAGES = [{ name: 'project', path: '' }, { name: 'answers', path: 'answers/' }, { name: 'workbench', path: 'workbench/' }]
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
    await page.waitForTimeout(40)
  }
  await page.evaluate(() => window.scrollTo(0, 0))
}

for (const site of PAGES) {
  for (const view of VIEWS) {
    const context = await browser.newContext({ viewport: { width: view.width, height: view.height }, colorScheme: view.colorScheme, reducedMotion: 'reduce' })
    const page = await context.newPage()
    const errors = []
    // On localhost the page looks for the live backend; a refused connection there just means it is not running.
    page.on('console', (m) => { if (m.type() === 'error' && !m.location().url.startsWith('http://127.0.0.1:8000')) errors.push(m.text()) })
    page.on('pageerror', (e) => errors.push(String(e)))
    await page.goto(ROOT + site.path, { waitUntil: 'networkidle' })
    await page.waitForSelector('main')
    await scrollThrough(page)

    if (view.name === 'desktop') {
      const text = await page.evaluate(() => {
        const clone = document.body.cloneNode(true)
        clone.querySelectorAll('[data-quote], script, style, noscript').forEach((n) => n.remove())
        const attrs = [...document.querySelectorAll('[aria-label], [placeholder], [title]')].filter((n) => !n.closest('[data-quote]'))
          .map((n) => [n.getAttribute('aria-label'), n.getAttribute('placeholder')].filter(Boolean).join(' '))
        return (clone.innerText ?? clone.textContent) + '\n' + attrs.join('\n') + '\n' + document.title
      })
      const hits = [...text.matchAll(DASHES)].map((m) => text.slice(Math.max(0, m.index - 40), m.index + 40).replace(/\s+/g, ' '))
      if (hits.length) {
        failures++
        console.log(`FAIL (${site.name}): ${hits.length} hyphen or dash characters in visible text:`)
        hits.slice(0, 20).forEach((h) => console.log(`  ...${h}...`))
      } else {
        console.log(`OK (${site.name}): no hyphens or dashes in visible text (quoted model answers excluded)`)
      }
    }

    await page.screenshot({ path: `screenshots/${site.name}_${view.name}.png`, fullPage: true })
    await page.screenshot({ path: `screenshots/${site.name}_${view.name}_top.png` })
    if (errors.length) { failures++; console.log(`FAIL (${site.name}, ${view.name}): browser errors:`, errors.slice(0, 5)) }
    else console.log(`OK (${site.name}, ${view.name}): no browser errors, screenshots saved`)
    await context.close()
  }
}

// The hero with motion on: after the opening, then at each stop of the scroll that quantizes the model.
for (const view of [VIEWS[0], VIEWS[2]]) {
  const context = await browser.newContext({ viewport: { width: view.width, height: view.height } })
  const page = await context.newPage()
  await page.goto(ROOT, { waitUntil: 'networkidle' })
  await page.waitForTimeout(500)
  await page.screenshot({ path: `screenshots/hero_${view.name}_opening.png` })
  for (const [p, name] of [[0, '16bit'], [0.48, '8bit'], [0.93, '4bit']]) {
    await page.evaluate((progress) => {
      const el = document.querySelector('.hero')
      window.scrollTo(0, el.offsetTop + progress * (el.offsetHeight - window.innerHeight))
    }, p)
    await page.waitForTimeout(p === 0 ? 2200 : 500)
    await page.screenshot({ path: `screenshots/hero_${view.name}_${name}.png` })
  }
  await context.close()
}
await browser.close()
console.log(failures ? `\n${failures} check(s) failed.` : '\nAll site checks passed.')
process.exit(failures ? 1 : 0)
