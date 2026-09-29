import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

// Builds one self-contained dist/index.html (served by backend/app.py, or opened directly).
export default defineConfig({
  plugins: [react(), viteSingleFile()],
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
});
