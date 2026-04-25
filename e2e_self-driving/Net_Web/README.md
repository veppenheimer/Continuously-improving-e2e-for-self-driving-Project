# Net_Web

`Net_Web` 是 `e2e_self-driving` 项目当前使用的 FastAPI 后端，负责项目管理、数据集上传、训练任务调度、进度查询、结果查看和单图推理对比。

当前 Web 训练入口只保留三种主模型架构：

- `legacy`：Legacy CNN
- `mobilenet_v2`：单帧 MobileNetV2
- `temporal3`：固定 3 帧输入的时序模型

训练流程仍然分为两段：

- `baseline`：只使用 A 域数据训练
- `augmented`：先生成 A→C 的域增强数据，再用 A + C 继续训练

不再支持旧的分类分支与改进分支，Web API、任务进度、结果页和对比推理都已移除对应字段。

## 环境准备

1. 进入 `Net_Web`
2. 创建虚拟环境
3. 安装依赖
4. 配置 `.env`
5. 启动 FastAPI 服务

示例：

```powershell
cd Net_Web
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

前端默认访问：

- API: `http://127.0.0.1:8000`
- WebSocket: `ws://127.0.0.1:8000`

健康检查：

```text
GET /health
```

## 训练任务说明

创建任务时需要提供：

- 项目 ID
- 数据集 A
- `modelVariant`
- 学习率、batch size、epochs
- 是否开启域增强
- 域增强相关 CycleGAN 参数

`temporal3` 在 Web 侧固定采用：

- `numFrames = 3`
- `frameStride = 1`

训练输出会写入 `Net_Web/data_storage/tasks/<task_id>/`，其中保留：

- `progress.json`
- `training_summary.json`
- 域增强对比所需的 `domain_aug_pairs.json`
- 任务对应的 `baseline.pth` / `augmented.pth`

checkpoint 元数据会写入：

- `modelVariant`
- `numFrames`
- `frameStride`
- `preprocess`

从而保证后续推理能够自动识别模型结构。

## 目录概览

- `app/main.py`：FastAPI 入口
- `app/routers/tasks.py`：训练任务 API
- `app/training_runner.py`：后台训练线程
- `app/services/inference.py`：单图推理与 checkpoint 加载
- `datasets.py` / `models.py`：Web 训练使用的数据集与模型封装

## 注意事项

- 训练中支持暂停、继续、终止
- 域增强会调用外部 CycleGAN 工程，请确保 `settings.cyclegan_project_root` 指向有效目录
- 历史旧任务如果缺少 `modelVariant`，前端会按“旧任务/未知架构”显示，而不会继续渲染已删除的旧竞赛字段
