import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [svelte(), tailwindcss()],
  server: {
    port: 5042,
    proxy: {
      '/api': { target: 'http://localhost:5041', changeOrigin: true },
      '/ws': { target: 'ws://localhost:5041', ws: true },
    },
  },
});
