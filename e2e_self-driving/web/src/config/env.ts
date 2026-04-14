/** 后端 HTTP 根路径，由 .env 中 VITE_API_BASE_URL 注入 */
export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "/api";

/** 可选 WebSocket 根（无则仅用轮询） */
export const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL?.replace(/\/$/, "") ?? "";
