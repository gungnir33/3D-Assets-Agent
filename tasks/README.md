# 3D Assets Agent 任务文档

本目录面向两类读者：维护和扩展项目的开发者，以及在本机调用 3D 生成能力的使用者。

## 文档入口

- [开发说明](DEVELOPMENT.md)：工程架构、环境搭建、模型管理、执行链路、测试、调试和发布流程。
- [使用说明](USAGE.md)：启动服务、检查模型、调用 Skill/API、查看任务产物、转换格式和排查问题。

## 项目定位

本项目是在单台 NVIDIA GPU 工作站上运行的本地 3D 资产生成服务：

```text
Codex
  → local-3d-agent Skill
  → 127.0.0.1:8080 FastAPI
  → HunyuanDiT / Hunyuan3D Shape / Hunyuan3D Paint
  → GLB（主要）/ OBJ（次要）
```

当前提供三条同步、串行的 GPU 工作流：

1. 文本 → 条件图 → 3D 网格 → 可选纹理。
2. 图片 → 3D 网格 → 可选纹理。
3. 现有网格 + 条件图 → 网格预处理 → 纹理 GLB。

## 重要边界

- 服务只绑定 `127.0.0.1`，不应直接暴露到局域网或公网。
- 模型、本地输入、生成资产和密钥不进入 Git。
- `assets/` 中除 `.gitkeep` 外的内容均被 Git 忽略。
- V0.1 不包含任务队列、Web UI、多 GPU、并发 GPU 推理或自动后台服务。
- 服务和 Skill 使用 Conda 环境 `hunyuan3d`（Python 3.10），不要使用 base Python 3.13。

## 已验证基线

当前部署证据位于项目根目录的 `deployment_manifest.json`。已验证硬件为 NVIDIA RTX 5880 Ada Generation，PyTorch `2.7.1+cu128`，Torch CUDA runtime `12.8`。

文档中的路径以本机部署根目录为准：

```text
/home/mcl/workspace/3D-Assets-Agent
```

