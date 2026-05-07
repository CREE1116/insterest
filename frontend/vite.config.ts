import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost',
        changeOrigin: true
      },
      '/profiles': {
        target: 'http://localhost',
        changeOrigin: true
      },
      '/uploads': {
        target: 'http://localhost',
        changeOrigin: true
      },
      '/outputs': {
        target: 'http://localhost',
        changeOrigin: true
      }
    }
  }
})
