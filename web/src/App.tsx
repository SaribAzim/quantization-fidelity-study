import { useEffect, useState } from 'react'
import { DataContext, loadSiteData, REPO, type SiteData } from './data'
import Hero from './sections/Hero'
import Question from './sections/Question'
import Rounding from './sections/Rounding'
import Showcase from './sections/Showcase'
import TryBest from './sections/TryBest'
import Results from './sections/Results'
import DidNotHelp from './sections/DidNotHelp'
import Method from './sections/Method'

function Topbar() {
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
        <a className="brand" href="#top">Quantization Fidelity<span className="brand-long"> Study</span></a>
        <nav className="navlinks" aria-label="Sections">
          <a href="#answers">Answers</a>
          <a href="#try">Try it</a>
          <a href="#results">Results</a>
          <a className="optional" href="#method">Method</a>
          <a className="optional" href={REPO}>Code</a>
        </nav>
      </div>
    </header>
  )
}

export default function App() {
  const [data, setData] = useState<SiteData | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    loadSiteData().then(setData).catch((e: Error) => setError(e.message))
  }, [])

  if (error) return <div className="loading" role="alert">The data for this page did not load. {error}. Reload the page to try again.</div>
  if (!data) return <div className="loading" aria-live="polite">Loading the results</div>

  return (
    <DataContext.Provider value={data}>
      <Topbar />
      <main id="top">
        <Hero />
        <Question />
        <Rounding />
        <Showcase />
        <TryBest />
        <Results />
        <DidNotHelp />
        <Method />
      </main>
      <footer className="footer">
        <div className="wrap footer-inner">
          <span>Built by Sarib Azim, on one RTX 3060 Ti with free tools.</span>
          <span><a href={REPO}>Source code and full results on GitHub</a></span>
          <span>Llama 3.1 is released by Meta under the Llama 3.1 Community License. No model weights are hosted here.</span>
        </div>
      </footer>
    </DataContext.Provider>
  )
}
