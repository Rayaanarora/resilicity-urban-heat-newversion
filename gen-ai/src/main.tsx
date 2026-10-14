import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// ?motion=off disables all animation (projector or screenshot mode).
if (new URLSearchParams(window.location.search).get('motion') === 'off') {
  document.documentElement.classList.add('rc-no-motion');
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
