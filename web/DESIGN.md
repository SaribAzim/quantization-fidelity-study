# Design notes

Written following `SKILL.md` (frontend design): plan the tokens first, review them against the brief, then build.

## Brief

- **Subject:** compressing Llama 3.1 8B so it fits one 8 GB graphics card, and measuring how faithfully the small version copies the original.
- **Audience:** recruiters and engineers.
- **Primary job:** within a minute, show that the best 4 bit model is 5.9 GB and answers like the original, then let people watch and test it.
- **Hard rules from the client:** no hyphens or dashes in any text; clean, professional, production grade; load and scroll animation; easy figures; a cream or beige theme; two pages.

## Pages

| Page | File | Content |
|---|---|---|
| The project | `index.html` | Pinned hero where scrolling quantizes the model, the question, the rounding explainer, results, what did not help, how it was built, a closing link to the test page |
| Model answers | `answers/index.html` | A profile with four small charts for the original, 8 bit and best 4 bit models; the side by side replay with highlighted differences and an agreement strip per answer; word similarity across all ten questions |
| Workbench | `workbench/index.html` | The author's testing page, linked only from the footer and marked noindex: recorded answers for every model, live questions to the 4 bit models when the local backend runs |

All pages share one script (`src/main.tsx`), which reads `data-page` on the body.

## Tokens

| Name | Light | Dark | Role |
|---|---|---|---|
| Cream | `#F3ECDF` | `#1C1915` | page background |
| Deep cream | `#EBE1CF` | `#15130F` | panel headers |
| Ivory | `#FBF7EF` | `#26221C` | figure and answer surfaces |
| Espresso | `#2A2520` | `#F1E9DB` | text |
| Taupe | `#6B6255` | `#AA9F8E` | secondary text, axes |
| Ink blue | `#2C4A7A` | `#8FA8D6` | the original model, negative weights, links |
| Sage | `#3D6B52` | `#8CC09F` | the best 4 bit model, recovery, fits |
| Mustard | `#C9962B` | `#DDB14F` | the 8 GB line, positive weights, highlights |

Data colours only: brick `#A8432C` (damage, does not fit), plum `#7B5279` (GPTQ), warm grey `#8A8273` (8 bit), berry `#9E3F55` (QLoRA).

**Type.** Archivo (variable weight and width) for headings, interface and every number, with tabular figures. Source Serif 4 for body text at 18 px, line height 1.6. Scale ratio 1.25.

**Layout.** Left aligned, 1200 px maximum, 16 px side gutter on phones. Text sits in a narrow column; figures run wide.

## Principles

1. One memorable thing: the pinned hero. On load the title letters rise and a real grid of weights develops like a photograph. Then the visitor's scroll quantizes the model: the grid sweeps from 16 to 8 to 4 bit and visibly bands into fewer colours, the headline narrows, and the size counts down past the 8 GB line.
2. Every number and figure is read from `public/data/*.json`, generated from the research results.
3. Motion only where it explains: the hero, the scroll driven rounding explainer, the answer replays, and one draw per chart. Reduced motion shows final states and the hero is not pinned.
4. Copy: plain words, sentence case, active verbs, buttons say what happens. No hyphens, dashes, middle dots or arrows. Model answers are quoted exactly as generated and are exempt.

## Review against the brief

Cream with a serif is a common default, so the page avoids its usual companions: no terracotta accent and no serif display face. Headings use a compressing grotesque that carries the idea of compression, and the accents (ink blue, sage, mustard) come from the data roles. Numbered markers appear only where the content is a real sequence (the rounding steps and the build pipeline).

## Checks

- `npm run build`: type checks and builds both pages.
- `npm run check` (with `npm run preview` running): Playwright opens both pages, fails on any hyphen or dash in visible text, labels or placeholders outside quoted model answers, fails on browser errors, and saves screenshots (desktop, tablet, phone, dark, and the hero at each scroll stage) to `screenshots/`.
