import { useEffect, useState } from 'react'
import { ANSWERS_PAGE, DataContext, loadSiteData, PROJECT_PAGE, REPO, WORKBENCH_PAGE, type SiteData } from './data'
import Hero from './sections/Hero'
import Question from './sections/Question'
import Rounding from './sections/Rounding'
import Results from './sections/Results'
import DidNotHelp from './sections/DidNotHelp'
import Method from './sections/Method'
import NextPage from './sections/NextPage'
import Answers from './sections/Answers'
import Workbench from './sections/Workbench'

export type Page = 'project' | 'answers' | 'workbench'

function Topbar({ page }: { page: Page }) {
  const [scrolled, setScrolled] = useState(false)
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 8)
    on()
    window.addEventListener('scroll', on, { passive: true })
    return () => window.removeEventListener('scroll', on)
  }, [])
  return (
    <header className={`topbar${scrolled ? ' scrolled' : ''}`}>
      <div className="wrap topbar-inner">
        <a className="brand" href={PROJECT_PAGE}>Quantization Fidelity<span className="brand-long"> Study</span></a>
        <nav className="navlinks" aria-label="Pages">
          <a href={PROJECT_PAGE} aria-current={page === 'project' ? 'page' : undefined}>The project</a>
          <a href={ANSWERS_PAGE} aria-current={page === 'answers' ? 'page' : undefined}>Model answers</a>
          {page === 'workbench' && <a href={WORKBENCH_PAGE} aria-current="page">Workbench</a>}
          <a className="optional" href={REPO}>Code</a>
        </nav>
      </div>
    </header>
  )
}

export default function App({ page }: { page: Page }) {
  const [data, setData] = useState<SiteData | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    loadSiteData().then(setData).catch((e: Error) => setError(e.message))
  }, [])

  if (error) return <div className="loading" role="alert">The data for this page did not load. {error}. Reload the page to try again.</div>
  if (!data) return <div className="loading" aria-live="polite">Loading the results</div>

  return (
    <DataContext.Provider value={data}>
      <Topbar page={page} />
      {page === 'project' && (
        <main>
          <Hero />
          <Question />
          <Rounding />
          <Results />
          <DidNotHelp />
          <Method />
          <NextPage />
        </main>
      )}
      {page === 'answers' && <main className="sub-page"><Answers /></main>}
      {page === 'workbench' && <main className="sub-page"><Workbench /></main>}
      <footer className="footer">
        <div className="wrap footer-inner">
          <span>Built on one RTX 3060 Ti with free tools.</span>
          <span><a href={REPO}>Source code and full results on GitHub</a></span>
          <span>Llama 3.1 is released by Meta under the Llama 3.1 Community License. No model weights are hosted here.</span>
          <span><a className="quiet-link" href={WORKBENCH_PAGE}>Workbench</a></span>
        </div>
      </footer>
    </DataContext.Provider>
  )
}
