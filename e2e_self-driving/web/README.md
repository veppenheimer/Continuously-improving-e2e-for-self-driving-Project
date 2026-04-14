# 端到端自动驾驶 · 训练控制台（前端）

基于需求文档实现的 Web 控制台：**React 18 + TypeScript + Vite**，UI 为 **Tailwind CSS + ShadCN 风格组件**，状态 **Zustand**，请求 **Axios**，曲线 **ECharts**。

## 环境准备（从零）

1. **安装 Node.js LTS**  
   - 打开 [https://nodejs.org/](https://nodejs.org/)，下载并安装 **LTS** 版本。  
   - 安装完成后，重新打开终端（PowerShell 或 CMD），执行：
     - `node -v`
     - `npm -v`  
     若都能显示版本号即成功。

2. **进入本项目前端目录**

```bash
cd web
```

3. **安装依赖**

```bash
npm install
```

4. **配置后端地址**

复制 `.env.example` 为 `.env`，修改 `VITE_API_BASE_URL` 为你的 Python 服务地址（不要末尾斜杠），例如：

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

开发阶段若需避免跨域，也可设为 `VITE_API_BASE_URL=/api`，并在启动 Vite 前设置代理目标（见 `vite.config.ts` 中 `server.proxy`），或在 shell 中：

```bash
set VITE_API_PROXY_TARGET=http://127.0.0.1:8000
npm run dev
```

5. **启动开发服务**

```bash
npm run dev
```

浏览器访问终端里提示的地址（一般为 `http://localhost:5173`）。

6. **生产构建**

```bash
npm run build
npm run preview
```

## 目录结构（摘要）

```
web/
├── docs/
│   └── BACKEND_API.md      # 与 Python 对接的接口约定
├── public/
├── src/
│   ├── api/                # Axios 实例、路径常量、按领域划分的 service
│   ├── components/         # UI 与布局、图表
│   ├── config/             # 环境变量读取
│   ├── hooks/              # 如训练进度轮询 / WebSocket
│   ├── lib/                # 工具函数
│   ├── pages/              # 各功能页面
│   ├── store/              # Zustand（登录态持久化）
│   ├── App.tsx
│   └── main.tsx
├── .env.example
├── package.json
├── tailwind.config.js
└── vite.config.ts
```

## 后端地址配置小结

| 变量 | 作用 |
|------|------|
| `VITE_API_BASE_URL` | 所有 HTTP 请求的 baseURL |
| `VITE_WS_BASE_URL` | 可选；若提供则连接 WebSocket 实时进度，否则自动轮询 `/tasks/:id/progress` |
| `VITE_API_PROXY_TARGET` | 仅当 base 为 `/api` 时，Vite 开发服务器转发目标 |

详细请求路径与 JSON 形状见 **`docs/BACKEND_API.md`**。你只需要让 Python 服务实现该文档中的接口；若路径或字段不同，优先改 `src/api/endpoints.ts` 与各 `src/api/services/*.ts`，业务页面尽量不改动。

## 功能与页面对应

| 模块 | 路由 |
|------|------|
| 登录 / 注册 | `/login`、`/register` |
| 历史任务 | `/` |
| 数据集 ZIP 上传 | `/datasets` |
| 新建训练（超参 + 域增强开关） | `/train/new` |
| 训练监控（曲线、暂停/终止） | `/tasks/:taskId/monitor` |
| 结果指标、推理对比、模型下载 | `/tasks/:taskId/results` |

## 后续拓展建议

- **新增强策略 / 模型结构**：在「新建训练」页增加表单项，扩展 `CreateTaskPayload` 与 `POST /tasks` body；后端识别新字段即可。  
- **接口版本化**：在 `src/api/endpoints.ts` 增加前缀如 `/v1`。  
- **权限角色**：在 `authStore` 与路由守卫中扩展 `user` 类型与受控路由。  
- **国际化**：引入 `i18next` 等，文案从页面抽离。

## 说明

当前仓库中的 **`Net/`** 为本地训练脚本，**不包含 HTTP 服务**。你需要自行用 FastAPI/Flask 等包装训练流程并实现 `docs/BACKEND_API.md` 中的接口，前端即可联调。
