import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource-variable/archivo/wdth.css'
import '@fontsource-variable/source-serif-4/opsz.css'
import './styles.css'
import App from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App page={document.body.dataset.page === 'test' ? 'test' : 'project'} />
  </StrictMode>,
)
