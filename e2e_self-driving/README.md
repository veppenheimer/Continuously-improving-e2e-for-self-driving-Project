一个端到端自动驾驶的简单实践

仅融合单传感器进行推理，输出为单维的舵机偏向角。

## Web 训练控制台

网页版训练与对比系统位于 **`web/`** 目录（React + Vite + TypeScript）。环境依赖、启动方式与后端接口约定见 [`web/README.md`](web/README.md) 与 [`web/docs/BACKEND_API.md`](web/docs/BACKEND_API.md)。

配套 **FastAPI 后端** 在 **`Net_Web/`**，与上述接口文档一致；启动说明见 [`Net_Web/README.md`](Net_Web/README.md)。
