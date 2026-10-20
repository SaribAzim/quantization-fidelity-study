import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The site is published at https://saribazim.github.io/quantization-fidelity-study/
export default defineConfig({
  plugins: [react()],
  base: '/quantization-fidelity-study/',
})
