import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv, type ProxyOptions } from "vite";
import react from "@vitejs/plugin-react";

const rootDir = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, rootDir, "");
  const apiTarget = env.VITE_API_PROXY_TARGET || "http://127.0.0.1:8000";
  const apiProxy: Record<string, string | ProxyOptions> = {
    "/api": {
      target: apiTarget,
      changeOrigin: true,
      ws: true,
      rewrite: (p) => p.replace(/^\/api/, ""),
    },
  };

  return {
    plugins: [react()],
    resolve: {
      alias: {
        "@": path.resolve(rootDir, "./src"),
      },
    },
    server: {
      host: "0.0.0.0",
      port: 5173,
      proxy: apiProxy,
    },
    preview: {
      host: "0.0.0.0",
      port: 4173,
      proxy: apiProxy,
    },
  };
});
