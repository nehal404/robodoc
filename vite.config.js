import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/robodoc/',
  server: {
    open: true,
    hmr: {
      overlay: false,
    },
  },
  build: {
    outDir: 'dist',
    copyPublicDir: true, // Ensure public/ is copied
  },
});