// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // Expose Vite server to network (e.g., for Docker)
    port: 5173, // Frontend runs on port 5173
    proxy: {
      '/phq9-chat': {
        target: 'http://localhost:8000', // Backend runs on port 8000
        changeOrigin: true, // Adjusts Host header to match target
        secure: false, // Disable SSL verification for local development
      },
      '/health': {
        target: 'http://localhost:8000', // Backend health endpoint on port 8000
        changeOrigin: true,
        secure: false,
      },
    },
  },
});