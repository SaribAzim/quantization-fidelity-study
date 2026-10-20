# Design notes

Written following `SKILL.md` (frontend design): plan the tokens first, review them against the brief, then build.

## Brief

- **Subject:** compressing Llama 3.1 8B so it fits one 8 GB graphics card, and measuring how faithfully the small version copies the original.
- **Audience:** recruiters and engineers.
- **Primary job:** within a minute, show that the best 4 bit model is 5.9 GB and answers like the original, then let people watch and try it.
- **Hard rules from the client:** no hyphens or dashes in any text; clean, professional, production grade; load and scroll animation; easy figures.

## Tokens

| Name | Light | Dark | Role |
|---|---|---|---|
| Porcelain | `#EDF0F4` | `#0F151C` | page background |
| Paper | `#FFFFFF` | `#17202A` | figure and answer surfaces |
| Graphite | `#1C2530` | `#E6EBF1` | text |
| Steel | `#5E6B7A` | `#93A1B1` | secondary text, axes, rules |
| Signal blue | `#1F57D6` | `#6F98F2` | the original model, links |
| Teal | `#0F8C7E` | `#3CC2B1` | the best 4 bit models, recovery |

Data colours only: ember `#C23B2C` (damage, negative results), limit amber `#E9A400` (the 8 GB line), violet `#7C5CC4` (GPTQ), slate `#6B7A8C` (8 bit), berry `#9C3B57` (QLoRA).

**Type.** Archivo (variable weight and width) for headings, interface and every number, with tabular figures. Source Serif 4 for body text at 18 px, line height 1.6, measure at most 68 characters. Scale ratio 1.25. The hero headline is set wide and compresses its width as the model is quantized: the type performs the subject.

**Layout.** Left aligned 12 column grid, 1200 px maximum. Text sits in a narrow column; figures run wide.

## Principles

1. One memorable thing: the hero, where a real slice of weights snaps to fewer colours while the headline compresses and the size drops below the 8 GB line. Everything else stays quiet.
2. Every number and figure is read from `public/data/*.json`, generated from the research results.
3. Motion only where it explains: the hero sequence, the scroll driven rounding explainer, answer replays the visitor starts, and one draw per chart. Reduced motion shows final states.
4. Copy: plain words, sentence case, active verbs, buttons say what happens. No hyphens, dashes, middle dots or arrows. Model answers are quoted exactly as generated and are exempt.

## Review against the brief

The first instinct was a warm off white page with a serif display face and monospace numbers, opened by a big number with stat cards. Each of these is a listed default, so each was replaced: the background is a cool porcelain, headings use a compressing grotesque that carries the idea of compression, numbers use tabular figures in the same family, and the hero is an interactive grid of real weights. Numbered markers appear only where the content is a real sequence (the rounding steps and the build pipeline).

## Checks

- `npm run build`: type checks and builds.
- `npm run check` (with `npm run preview` running): Playwright opens the site, fails on any hyphen or dash in visible text outside quoted model answers, and saves screenshots at desktop, tablet, phone and dark mode to `screenshots/`.
