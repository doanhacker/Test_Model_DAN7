import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
import tailwind from '@tailwindcss/vite';
export default defineConfig({plugins:[react(),tailwind()],server:{proxy:{'/api':'http://127.0.0.1:8000'}},build:{rollupOptions:{input:{index:'index.html','image-lab':'image-lab.html'},output:{entryFileNames:chunk=>chunk.name==='image-lab'?'assets/image-lab.js':'assets/[name]-[hash].js',assetFileNames:asset=>asset.names?.some(n=>n==='image-lab.css')?'assets/image-lab.css':'assets/[name]-[hash][extname]'}}}});
