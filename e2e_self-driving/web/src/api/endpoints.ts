/**
 * 集中定义路径，后端调整时只改此文件（及 types）。
 * 默认 REST 风格；若你的 FastAPI 路径不同，在此统一替换。
 */
export const paths = {
  register: "/auth/register",
  login: "/auth/login",
  me: "/auth/me",

  datasets: "/datasets",
  datasetUpload: "/datasets/upload",
  dataset: (id: string) => `/datasets/${id}`,

  tasks: "/tasks",
  task: (id: string) => `/tasks/${id}`,
  taskProgress: (id: string) => `/tasks/${id}/progress`,
  taskPause: (id: string) => `/tasks/${id}/pause`,
  taskStop: (id: string) => `/tasks/${id}/stop`,
  taskResults: (id: string) => `/tasks/${id}/results`,
  taskInferCompare: (id: string) => `/tasks/${id}/infer/compare`,
  taskDownload: (id: string) => `/tasks/${id}/download`,
  taskDomainAugPairs: (id: string) => `/tasks/${id}/domain-aug/pairs`,
  taskDomainAugImage: (id: string) => `/tasks/${id}/domain-aug/image`,

  /** WebSocket: `${WS_BASE_URL}/tasks/${id}/stream` */
  taskStream: (id: string) => `/tasks/${id}/stream`,
} as const;
