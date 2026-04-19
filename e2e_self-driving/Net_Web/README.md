# Net_Web — 端到端训练 HTTP 服务（FastAPI）

在 **`Net/`** 原有 `models.py`、`datasets.py`、`utils.py` 能力之上，复制并扩展为可运行的 **FastAPI** 后端，接口与仓库内 **`web/docs/BACKEND_API.md`** 及前端 `web/` 一致。

## 功能概览

- **认证**：注册 / 登录（JWT）、`/auth/me`
- **数据集**：ZIP 上传、解压、按 `序号_转向角.jpg` 解析、80/20 划分、`train.txt` / `val.txt`
- **训练**：后台线程跑 PyTorch；**基准模型** + 可选 **域增强模型**（ColorJitter / 随机高斯模糊等）；每轮 **训练 Loss + 验证 Loss**；内存中曲线数据供轮询 / WebSocket
- **控制**：暂停 / 继续（同一 `pause` 接口切换）、终止
- **结果**：指标 JSON、对比推理、`.pth` 下载
- **WebSocket**：`GET` 同参 `?token=` → `/tasks/{id}/stream`，推送与 `/progress` 相同结构的 JSON

数据与数据库默认写在 **`Net_Web/data_storage/`**（可在 `.env` 中改）。

## 环境（Windows / PyCharm 均可）

1. **Python 3.10+**（推荐 3.11）
2. 建议虚拟环境：

```powershell
cd Net_Web
python -m venv .venv
.\.venv\Scripts\activate
```

3. 安装依赖（CPU 版 PyTorch 可直接 pip；**GPU** 请先到 [pytorch.org](https://pytorch.org) 按 CUDA 版本安装 `torch` / `torchvision`，再装其余包）：

```powershell
pip install -r requirements.txt
```

4. 配置环境变量：复制 `.env.example` 为 `.env`，至少修改 **`SECRET_KEY`**。

5. 启动服务（需在 **`Net_Web`** 目录下执行，保证能 `import models`）：

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

6. 浏览器或前端将 **`VITE_API_BASE_URL`** 设为 `http://127.0.0.1:8000`；若使用 WebSocket，**`VITE_WS_BASE_URL`** 设为 `ws://127.0.0.1:8000`。

健康检查：`GET http://127.0.0.1:8000/health`

## 目录说明

| 路径 | 说明 |
|------|------|
| `models.py` / `datasets.py` / `utils.py` | 与原 Net 一致的网络与数据管线（`datasets` 中行格式为 `路径 角度`，支持路径中含空格时用最后一个空格分隔） |
| `app/main.py` | FastAPI 应用入口 |
| `app/routers/` | `auth`、`datasets`、`tasks` 路由 |
| `app/training_runner.py` | 训练循环与检查点保存 |
| `app/services/` | ZIP ingest、推理预处理 |
| `create_data_lists.py` | 离线生成本地列表（可选；网页上传不需要） |

## 竞赛分类模型说明

当前 Web 端已经接入 `e2e_competition/Net_class` 的高精度分类链路。

### 训练任务中的 `competitionClass`

当任务启用分类竞赛模型后：

- Web 后端会调用 `e2e_competition/Net_class/train.py`
- 分类模型输出为 `9` 个类别 logits 加 `1` 个残差，共 `10` 维
- `competitionClass.steeringError` 表示真实角度误差，即 `Val_Angle_MAE`
- TensorBoard 中会记录：
  - `CE_Loss`
  - `Val_CE_Loss`
  - `Train_Acc`
  - `Val_Acc`
  - `Train_Angle_MAE`
  - `Val_Angle_MAE`

### 比较推理中的 `competitionClass`

`/tasks/{id}/infer/compare` 在加载分类竞赛模型时，不再使用“`argmax + 固定角度表`”的硬解码方式，而是复用 `Net_class/steering_config.py` 中的 `decode_output()`：

- 先确定主类别所在转向组
- 仅在组内做 softmax 加权
- 再叠加残差头输出的微小偏移

这样能让分类模型在不增加太多部署复杂度的前提下，输出更连续、更贴近真实场地需求的转向角。

### 与 `competitionLite` 的关系

- `competitionClass` 对应 `e2e_competition/Net_class`
- `competitionLite` 对应 `e2e_competition/Net_improve`

两条链路分别维护；本次高精度分类改造不影响 `competitionLite`。

## 注意事项

- **进程重启**后内存中的训练曲线会丢失；任务状态与结果在 SQLite 中仍可查，但进度曲线可能为空直至新任务产生数据。
- 训练占用 GPU/CPU 资源较大；单机可同时跑多个任务，请自行控制并发。
- 暂停在 **epoch 边界**生效；终止会在当前 epoch 内尽快跳出。
- 分类竞赛模型依赖 `e2e_competition/Net_class/steering_config.py`；如果你调整该目录下的解码规则，Web 端会同步复用。

## 与 `Net/` 的关系

原 **`Net/train.py`** 仍为脚本式入口；日常联调前端请使用本 **`Net_Web`** 服务。若需命令行单机训练，可继续使用原 `Net` 目录或参考其中脚本。