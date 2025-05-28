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
    copyPublicDir: true,
    rollupOptions: {
      onLog(level, log, handler) {
        if (log.message.includes('copying')) {
          console.log(log.message); // Log file copying
        }
        handler(level, log);
      },
    },
  },
});