# 3D Assets Agent 使用说明

## 1. 快速开始

以下命令假定项目位于：

```text
/home/mcl/workspace/3D-Assets-Agent
```

先检查环境和模型：

```bash
cd /home/mcl/workspace/3D-Assets-Agent
conda run -n hunyuan3d python scripts/check_environment.py
conda run -n hunyuan3d python scripts/check_models.py
```

启动本地服务：

```bash
./scripts/start_server.sh
```

服务在前台运行并绑定：

```text
http://127.0.0.1:8080
```

不要关闭这个终端。另开一个终端检查：

```bash
curl -fsS http://127.0.0.1:8080/health
```

正常响应示例：

```json
{
  "status": "ok",
  "cuda": true,
  "gpu": "NVIDIA RTX 5880 Ada Generation",
  "models": {
    "hunyuan3d_shape": "ready",
    "hunyuan3d_paint": "ready",
    "hunyuandit": "ready"
  }
}
```

`/health` 不加载大型模型，第一次生成仍需等待模型加载。

## 2. 使用前检查

### 2.1 GPU

```bash
nvidia-smi
```

确认：

- GPU 可见。
- 没有未知的大显存进程。
- 可用显存足以执行目标工作流。

不要自动终止其他用户或未知进程。

### 2.2 主模型

```bash
conda run -n hunyuan3d python scripts/check_models.py
```

可能状态：

- `READY`：可直接使用。
- `MISSING`：模型目录不存在。
- `PARTIAL`：模型目录不完整。

显式补齐：

```bash
conda run -n hunyuan3d python scripts/download_models.py --model hunyuan3d
conda run -n hunyuan3d python scripts/download_models.py --model hunyuandit
```

也可以一次补齐两套主模型：

```bash
conda run -n hunyuan3d python scripts/download_models.py --all
```

下载保存到 manifest 指定的固定目录。项目不会强制修改 Hugging Face endpoint 或代理设置。

### 2.3 rembg

透明 PNG 可以直接进入 Image-to-3D。不透明 PNG/JPG/WebP 会使用 rembg 去背景。

默认目录：

```text
/home/mcl/models/rembg/u2net.onnx
```

如果文件缺失：

- `models.allow_download=true`：首次遇到不透明图片时允许 rembg 下载。
- `models.allow_download=false`：请求失败并明确提示模型缺失。

## 3. 推荐方式：Codex Skill

Skill 应以符号链接安装：

```bash
mkdir -p /home/mcl/.agents/skills
ln -sfn \
  /home/mcl/workspace/3D-Assets-Agent/skill/local-3d-agent \
  /home/mcl/.agents/skills/local-3d-agent
```

确认链接：

```bash
readlink -f /home/mcl/.agents/skills/local-3d-agent
```

Skill 不会自动启动服务。服务未运行时输出：

```text
LOCAL_3D_SERVER_NOT_RUNNING
```

并提示执行：

```bash
/home/mcl/workspace/3D-Assets-Agent/scripts/start_server.sh
```

### 3.1 Skill 健康检查

```bash
cd /home/mcl/workspace/3D-Assets-Agent
conda run -n hunyuan3d python skill/local-3d-agent/scripts/health.py
```

### 3.2 文本生成 3D

默认生成带纹理 GLB：

```bash
conda run -n hunyuan3d python \
  skill/local-3d-agent/scripts/generate_text.py \
  "a futuristic industrial quadruped robot"
```

只生成几何，不运行 Paint：

```bash
conda run -n hunyuan3d python \
  skill/local-3d-agent/scripts/generate_text.py \
  "a futuristic industrial quadruped robot" \
  --no-texture
```

指定参数和复制目录：

```bash
conda run -n hunyuan3d python \
  skill/local-3d-agent/scripts/generate_text.py \
  "industrial inspection robot" \
  --seed 20260910 \
  --shape-steps 60 \
  --face-count 30000 \
  --output-dir /home/mcl/output/robot
```

### 3.3 图片生成 3D

```bash
conda run -n hunyuan3d python \
  skill/local-3d-agent/scripts/generate_image.py \
  /absolute/path/input.png
```

几何模式：

```bash
conda run -n hunyuan3d python \
  skill/local-3d-agent/scripts/generate_image.py \
  /absolute/path/input.png \
  --no-texture \
  --seed 12345 \
  --shape-steps 50
```

图片建议：

- 主体居中，轮廓清晰。
- 尽量单主体，避免遮挡。
- 优先透明背景 PNG。
- 不透明图片会触发 rembg。
- 支持 `.png`、`.jpg`、`.jpeg`、`.webp`，最大 100 MiB。

### 3.4 给现有网格生成纹理

```bash
conda run -n hunyuan3d python \
  skill/local-3d-agent/scripts/texture.py \
  /absolute/path/model.glb \
  /absolute/path/condition.png \
  --face-count 40000 \
  --seed 12345
```

mesh 支持 `.glb`、`.gltf`、`.obj`、`.ply`、`.stl`，最大 100 MiB。Paint 前会清理浮岛、退化面并将面数约束到请求上限。

条件图应与模型目标外观一致；它不是普通贴图文件，而是 Paint 的视觉条件。

### 3.5 格式转换

GLB 转 OBJ：

```bash
conda run -n hunyuan3d python \
  skill/local-3d-agent/scripts/convert.py \
  /absolute/path/model.glb \
  /absolute/path/model.obj
```

当前 helper 不支持 FBX。GLB 是首选格式。

## 4. 直接调用 HTTP API

所有请求均为同步调用。请将客户端超时设置为至少 900 秒。

### 4.1 公共参数

| 字段 | 类型 | 默认值 | 范围/说明 |
|---|---|---:|---|
| `texture` | boolean | `true` | 是否运行 Paint |
| `seed` | integer | `12345` | 0 到 `2^63-1` |
| `format` | string | `glb` | `glb`、`obj`、`fbx`；当前 FBX 不可用 |
| `face_count` | integer | `40000` | 100 到 1,000,000 |
| `shape_steps` | integer | `50` | 1 到 200 |

### 4.2 Text-to-3D

```bash
curl --max-time 900 \
  -X POST http://127.0.0.1:8080/generate/text \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "industrial inspection robot",
    "texture": true,
    "seed": 12345,
    "format": "glb",
    "face_count": 40000,
    "shape_steps": 50
  }'
```

### 4.3 Image-to-3D

服务只接受本机绝对路径，不接受文件上传：

```bash
curl --max-time 900 \
  -X POST http://127.0.0.1:8080/generate/image \
  -H 'Content-Type: application/json' \
  -d '{
    "image": "/absolute/path/input.png",
    "texture": true,
    "seed": 12345,
    "format": "glb",
    "face_count": 40000,
    "shape_steps": 50
  }'
```

### 4.4 Existing Mesh-to-Texture

```bash
curl --max-time 900 \
  -X POST http://127.0.0.1:8080/generate/texture \
  -H 'Content-Type: application/json' \
  -d '{
    "mesh": "/absolute/path/model.glb",
    "condition_image": "/absolute/path/condition.png",
    "texture": true,
    "seed": 12345,
    "format": "glb",
    "face_count": 40000,
    "shape_steps": 50
  }'
```

### 4.5 成功响应

```json
{
  "job_id": "20260910_120000_ab12cd34",
  "file": "/home/mcl/workspace/3D-Assets-Agent/assets/20260910_120000_ab12cd34/model.glb",
  "type": "glb",
  "metadata": "/home/mcl/workspace/3D-Assets-Agent/assets/20260910_120000_ab12cd34/metadata.json"
}
```

### 4.6 校验和错误响应

字段错误、文件不存在、扩展名不支持或超过 100 MiB 时，FastAPI 返回 HTTP 422。

CUDA OOM 返回 HTTP 503：

```json
{
  "error": {
    "code": "CUDA_OUT_OF_MEMORY",
    "message": "...",
    "details": {
      "allocated_bytes": 5985050112,
      "reserved_bytes": 0,
      "total_bytes": 52763648000
    }
  }
}
```

一般推理错误使用 `GENERATION_FAILED`。查看响应对应任务目录内的 `metadata.json`。

## 5. Smoke Test

服务启动后，可以执行单条带纹理 smoke test。

文本：

```bash
conda run -n hunyuan3d python scripts/smoke_test.py \
  --prompt "a small industrial robot"
```

图片：

```bash
conda run -n hunyuan3d python scripts/smoke_test.py \
  --image /absolute/path/input.png
```

`smoke_test.py` 先检查 health，再发送一个 900 秒超时的同步请求。

## 6. 参数选择建议

### texture

- `true`：生成带纹理 GLB，需要加载 Paint，时间和显存更高。
- `false`：只生成几何，适合快速验证 Shape、后续自行制作材质或先检查拓扑。

Skill 的默认值是带纹理；使用 `--no-texture` 关闭。

### seed

固定 seed 有助于复现。要比较参数影响，应保持 prompt、输入图、模型 revision、Torch 和 seed 不变。不同硬件或依赖版本仍可能产生差异。

### shape_steps

- 较少步骤通常更快。
- 默认 50 是已验证基线。
- 允许范围 1–200。
- 增加步骤不保证线性提升质量。

### face_count

- 主要约束 Paint 前网格面数。
- 默认 40,000 已在 RTX 5880 Ada 上验证。
- 降低可减少 Paint 开销，但会损失几何细节。
- 提高会增加处理成本，不一定改善纹理质量。

### format

- 优先 `glb`：单文件、可嵌入纹理、验收覆盖最完整。
- `obj`：次要交换格式，可能附带额外材质或纹理文件。
- `fbx`：当前版本不支持，不要在生产请求中使用。

## 7. 任务产物

任务位于：

```text
/home/mcl/workspace/3D-Assets-Agent/assets/<job-id>/
```

典型 Text-to-3D：

```text
condition.png
raw_mesh.glb
processed_mesh.glb   # texture=true 时
model.glb
metadata.json
generation.log
```

典型 Image-to-3D：

```text
input.png
raw_mesh.glb
processed_mesh.glb   # texture=true 时
model.glb
metadata.json
generation.log
```

Existing Mesh-to-Texture 不复制外部输入 mesh 和条件图，而是在 metadata 中记录其绝对路径。

### metadata 重点字段

```json
{
  "status": "SUCCEEDED",
  "seed": 12345,
  "shape_steps": 50,
  "face_count": 40000,
  "device": "cuda",
  "dtype": "float16",
  "models": {
    "hunyuan3d": {
      "path": "/home/mcl/models/hunyuan3d/Hunyuan3D-2",
      "revision": "...",
      "downloaded": false
    }
  },
  "peak_vram_bytes": 15554407936,
  "generation_duration_seconds": 36.5,
  "output": "/absolute/path/model.glb"
}
```

`generation.log` 当前仅记录任务开始行。完整运行结果以 metadata 和服务前台日志为准。

## 8. 查看和验证 GLB

可以使用支持 glTF 2.0 的工具打开 `model.glb`，例如 Blender、MeshLab、FreeCAD 或可信的本地 glTF viewer。不要把私人模型上传到公共在线查看器。

用 trimesh 做结构检查：

```bash
conda run -n hunyuan3d python -c \
  "import trimesh; p='/absolute/path/model.glb'; m=trimesh.load(p, force='mesh'); print(len(m.vertices), len(m.faces), m.bounds)"
```

使用项目验证器检查普通 GLB：

```bash
conda run -n hunyuan3d python -c \
  "from pathlib import Path; from local_3d_agent.mesh.validate import validate_mesh; print(validate_mesh(Path('/absolute/path/model.glb')))"
```

检查带纹理 GLB：

```bash
conda run -n hunyuan3d python -c \
  "from pathlib import Path; from local_3d_agent.mesh.validate import validate_mesh; print(validate_mesh(Path('/absolute/path/model.glb'), require_texture=True))"
```

## 9. 停止服务

服务是前台进程。在运行 `scripts/start_server.sh` 的终端按 `Ctrl+C`，等待 Uvicorn 输出 shutdown complete。

Skill 不会留下后台 daemon。不要用粗暴的全局 `killall python` 停止服务。

## 10. 故障排查

### LOCAL_3D_SERVER_NOT_RUNNING

原因：Skill 无法连接 `127.0.0.1:8080`。

处理：

```bash
/home/mcl/workspace/3D-Assets-Agent/scripts/start_server.sh
```

然后重新运行 `health.py`。

### health 为 degraded 或 cuda=false

```bash
nvidia-smi
conda run -n hunyuan3d python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

确认使用的是 `hunyuan3d` 环境，而不是 base Python。

### 模型 MISSING/PARTIAL

```bash
conda run -n hunyuan3d python scripts/check_models.py
```

允许联网时执行对应 `download_models.py` 命令。不要把 repo id 直接替代本地路径传给生产 pipeline。

### rembg model is missing

输入是不透明图片且下载被关闭。处理方式之一：

- 将 `LOCAL_3D_MODELS__ALLOW_DOWNLOAD=true` 后重启服务，允许首次补齐。
- 预先将 `u2net.onnx` 放入 `/home/mcl/models/rembg`。
- 改用已正确去背景的透明 PNG。

### CUDA_OUT_OF_MEMORY

1. 查看错误 details 和 `nvidia-smi`。
2. 确认没有同时运行 GPU E2E 和 API 服务任务。
3. 降低 `face_count`。
4. 使用 `texture=false` 或 `--no-texture`。
5. 正常重启本项目服务以释放自身 pipeline。

不要让 Agent 自动杀死其他 GPU 进程。

### 请求长时间无响应

生成是同步的，第一次加载模型和 Paint 可能耗时较长。确认客户端 timeout ≥ 900 秒，并查看服务前台是否正在采样或加载权重。GPU 锁会让后来的请求等待前一个请求完成。

### 图片路径返回 422

检查：

- 使用绝对路径。
- 文件存在且为普通文件。
- 后缀为 PNG/JPG/JPEG/WebP。
- 文件不超过 100 MiB。
- 服务进程用户有读取权限。

### mesh 路径返回 422

检查：

- 使用绝对路径。
- 后缀为 GLB/GLTF/OBJ/PLY/STL。
- 文件不超过 100 MiB。
- mesh 确实包含有效几何。

### Paint 成功但 GLB 没有材质

服务会把缺少 UV、material 或实际 texture image 的输出判为失败。查看 `processed_mesh.glb`、条件图和服务日志，不能只检查最终文件大小。

### FBX 失败

当前版本没有完成 FBX 转换链路。使用 GLB，或先导出 GLB 后在 Blender 中手动转换。

### custom pipeline / trust_remote_code 错误

```bash
cd /home/mcl/workspace/3D-Assets-Agent
./scripts/apply_upstream_patches.sh
```

然后正常重启服务。

## 11. 安全与数据管理

- API 仅供本机使用，不要把 host 改成 `0.0.0.0`。
- 当前 API 没有认证、上传接口或请求体大小网关。
- 不要在 prompt、路径、metadata 或日志中放入 token。
- 不要提交 `assets/`、`.env`、模型目录或私人图片。
- 删除任务前确认路径，避免对项目根目录或模型根目录执行递归删除。

## 12. 当前能力边界

当前没有：

- Web UI。
- 异步任务查询。
- 多请求 GPU 并发。
- 自动后台启动和进程守护。
- LAN/公网 API。
- 可用的 FBX 自动转换。
- Blender 自动编辑。
- 多 GPU 调度。

需要上述能力时，应先扩展设计、安全边界和测试，而不是直接修改启动参数。

