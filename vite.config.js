import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/robodoc/', // Match your repository name
  server: {
    open: true,
  },
  build: {
    outDir: 'dist',
  },
});