import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The site is published at https://saribazim.github.io/quantization-fidelity-study/
// Three pages: the project (index.html), the model answers (answers/) and the author's workbench (workbench/).
export default defineConfig({
  plugins: [react()],
  base: '/quantization-fidelity-study/',
  build: {
    rollupOptions: {
      input: { project: 'index.html', answers: 'answers/index.html', workbench: 'workbench/index.html' },
    },
  },
})
