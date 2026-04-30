# 后端接口约定（供 Python 实现）

前端默认请求根路径由环境变量 `VITE_API_BASE_URL` 指定。认证方式为 **`Authorization: Bearer <token>`**（注册/登录除外）。

以下 JSON 字段名与 `src/api/types.ts` 一致；若你的 FastAPI/Pydantic 使用 snake_case，可在后端做别名或在 `src/api/services/*` 里做一次映射。

## 认证

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/auth/register` | body: `{ "username", "password", "email?" }` → `{ "token", "user" }` |
| POST | `/auth/login` | body: `{ "username", "password" }` → `{ "token", "user" }` |
| GET | `/auth/me` | 需 Bearer，返回 `User` |

`User`: `{ "id": string, "username": string, "email?": string }`

## 数据集

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/datasets` | 返回 `DatasetItem[]` |
| POST | `/datasets/upload` | `multipart/form-data`：`file`（ZIP），可选 `name` → 返回 `DatasetItem` |
| DELETE | `/datasets/{id}` | 删除数据集（若已被训练任务引用，应返回 400） |

`DatasetItem`: `{ "id", "name", "imageCount?", "createdAt" }`（`createdAt` 建议 ISO8601 字符串）

## 训练任务

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/tasks` | 当前用户的 `TrainingTaskSummary[]` |
| GET | `/tasks/{id}` | 单个任务详情 |
| POST | `/tasks` | body 见下 → 创建任务并返回摘要 |
| GET | `/tasks/{id}/progress` | 训练进度与曲线数据 |
| POST | `/tasks/{id}/pause` | 暂停 |
| POST | `/tasks/{id}/stop` | 终止 |
| GET | `/tasks/{id}/results` | 完成后指标 |
| POST | `/tasks/{id}/infer/compare` | `multipart`：`file`（图像）→ 对比推理结果 |
| GET | `/tasks/{id}/download` | query: `stage=baseline\|augmented`，返回模型文件流 |

### POST `/tasks` body

```json
{
  "datasetId": "string",
  "learningRate": 0.0001,
  "batchSize": 32,
  "epochs": 100,
  "domainAugmentation": false
}
```

### GET `/tasks/{id}/progress` 响应 `TaskProgress`

- `status`: `pending` \| `running` \| `paused` \| `completed` \| `failed` \| `stopped`
- `currentEpoch`, `totalEpochs`: number
- `baseline`: `{ "trainLossSeries": LossPoint[], "valLossSeries": LossPoint[] }`
- `augmented?`: 同结构（未开启域增强可省略或给空数组）
- `LossPoint`: `{ "epoch": number, "trainLoss": number, "valLoss": number }`

### GET `/tasks/{id}/results` 响应 `TaskResultSummary`

- `baseline`: `{ "finalTrainLoss", "finalValLoss", "steeringError" }`
- `augmented?`: 同上

### POST `/tasks/{id}/infer/compare` 响应

```json
{
  "baselineSteering": 0.12,
  "augmentedSteering": 0.08
}
```

未训练增强模型时可省略 `augmentedSteering`。

## WebSocket（可选）

若设置 `VITE_WS_BASE_URL`（例如 `ws://127.0.0.1:8000`），前端会连接：

`{VITE_WS_BASE_URL}/tasks/{id}/stream?token=<jwt>`

服务端推送的每条消息为 **JSON 字符串**，解析后结构与 `TaskProgress` 相同。

## 错误格式

建议 HTTP 4xx/5xx 返回 JSON：`{ "detail": "人类可读说明" }` 或 `{ "message": "..." }`，前端会用于 Toast 提示。
