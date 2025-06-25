import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';
import electron from 'vite-plugin-electron';

export default defineConfig({
  base: './', 
  plugins: [
    react(),
    electron({
      entry: 'electron/main.ts',
    }),
  ],
  resolve: {
    alias: [
      { find: '@', replacement: resolve(__dirname, 'src') },
      { find: '@components', replacement: resolve(__dirname, 'src/components') },
      { find: '@pages', replacement: resolve(__dirname, 'src/pages') },
      { find: '@hooks', replacement: resolve(__dirname, 'src/hooks') },
      { find: '@utils', replacement: resolve(__dirname, 'src/utils') }
    ]
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    manifest: true
  },
  server: {
    port: 3000,
    strictPort: true,
  },
  preview: {
    port: 3000,
    strictPort: true,
  }
});
