import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The site is published at https://saribazim.github.io/quantization-fidelity-study/
// Two pages: the project (index.html) and the model tester (test/index.html).
export default defineConfig({
  plugins: [react()],
  base: '/quantization-fidelity-study/',
  build: {
    rollupOptions: {
      input: { project: 'index.html', test: 'test/index.html' },
    },
  },
})
