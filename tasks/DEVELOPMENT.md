# 3D Assets Agent 开发说明

## 1. 文档目的

本文说明如何理解、维护、测试和发布 3D Assets Agent。内容以当前仓库实现为准，不把 V0.2 设想当作已实现能力。

项目根目录：

```text
/home/mcl/workspace/3D-Assets-Agent
```

上游源码目录：

```text
/home/mcl/workspace/hunyuan3d/Hunyuan3D-2
```

模型目录：

```text
/home/mcl/models/hunyuan3d/Hunyuan3D-2
/home/mcl/models/hunyuan3d/HunyuanDiT-v1.1-Diffusers-Distilled
/home/mcl/models/rembg
```

模型权重和 Hunyuan3D 上游源码不属于本项目 Git 仓库。

## 2. 系统架构

```text
Codex / 本地脚本 / curl
              │
              ▼
       local-3d-agent Skill
              │ HTTP，timeout=900s
              ▼
   FastAPI 127.0.0.1:8080
              │
              ▼
       单 GPU 任务互斥锁
              │
        GenerationBackend
       ┌──────┼─────────┐
       ▼      ▼         ▼
 HunyuanDiT  Shape     Paint
       │      │         ▲
       └─condition.png  │
              └─mesh preprocessing
                         │
                         ▼
                   validated GLB
```

组件责任边界：

- `config.py`：读取 YAML 和环境变量，执行严格配置校验。
- `model_resolver.py`：统一验证模型目录、补齐主模型并返回固定本地路径。
- `model_manager.py`：延迟加载 Shape、HunyuanDiT、Paint，控制释放和 CUDA cache 清理。
- `pipelines/`：封装单一推理阶段，不自行决定模型下载位置。
- `mesh/`：网格预处理、格式导出和最终产物验证。
- `service/`：API schema、GPU 锁、错误映射、metadata 和工作流编排。
- `skill/local-3d-agent/`：薄 HTTP 客户端，不加载模型、不修改环境、不管理后台进程。

## 3. 代码目录

```text
3D-Assets-Agent/
├── config/
│   ├── default.yaml
│   └── model_manifest.yaml
├── src/local_3d_agent/
│   ├── config.py
│   ├── model_manager.py
│   ├── model_resolver.py
│   ├── metadata.py
│   ├── mesh/
│   ├── pipelines/
│   └── service/
├── scripts/
├── skill/local-3d-agent/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── gpu/
├── docs/
├── tasks/
└── assets/
```

`assets/` 是运行时目录。不要在测试或文档提交中加入其中的 GLB、PNG、metadata 或用户输入。

## 4. 开发环境

### 4.1 基础要求

- Ubuntu Linux。
- NVIDIA 驱动可通过 `nvidia-smi` 访问。
- Conda 环境名固定为 `hunyuan3d`。
- Python `3.10.x`。
- PyTorch CUDA 构建可识别实际 GPU。
- 足够磁盘空间存放 Hunyuan3D 和 HunyuanDiT 权重。

当前已验证组合：

```text
Python              3.10.21
PyTorch             2.7.1+cu128
Torch CUDA runtime  12.8
NVIDIA Driver       575.64.05
GPU                  RTX 5880 Ada Generation, 48 GiB
```

驱动中 `CUDA Version` 表示驱动最高兼容能力，不要求与 `torch.version.cuda` 字面相同。

### 4.2 创建环境

```bash
conda create -n hunyuan3d python=3.10 -y
cd /home/mcl/workspace/3D-Assets-Agent
conda run -n hunyuan3d python -m pip install -e '.[test]'
```

项目的直接依赖声明在 `pyproject.toml`；当前部署的完整版本快照位于 `requirements.lock.txt`。不要用 base Python 运行服务。

Hunyuan3D 上游还拥有自己的依赖。首次部署时应在同一个 Conda 环境中安装上游要求，并确保没有覆盖已经验证可工作的 Torch CUDA 版本。安装后执行：

```bash
conda run -n hunyuan3d python -m pip check
```

### 4.3 CUDA Extension

编译前记录环境：

```bash
nvcc --version
conda run -n hunyuan3d python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

扩展位于上游仓库：

```text
hy3dgen/texgen/custom_rasterizer
hy3dgen/texgen/differentiable_renderer
```

应从各自源码目录执行安装。例如：

```bash
cd /home/mcl/workspace/hunyuan3d/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
conda run --no-capture-output -n hunyuan3d python setup.py install

cd /home/mcl/workspace/hunyuan3d/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
conda run --no-capture-output -n hunyuan3d python setup.py install
```

Ada 架构通常对应 `sm_89`。仅在构建系统没有正确检测时设置 `TORCH_CUDA_ARCH_LIST=8.9`，不要无条件覆盖已有配置。

加载扩展进行诊断时先导入 Torch，避免动态链接器找不到 `libc10.so`：

```bash
conda run -n hunyuan3d python -c "import torch; import custom_rasterizer_kernel; import mesh_processor; print('OK')"
```

### 4.4 上游兼容补丁

当前 Diffusers 版本加载 Paint 的本地自定义 pipeline 时需要显式信任本地代码。补丁脚本是幂等的：

```bash
cd /home/mcl/workspace/3D-Assets-Agent
./scripts/apply_upstream_patches.sh
```

它只修改独立上游工作树的 `hy3dgen/texgen/utils/multiview_utils.py`，不会复制上游源码到项目仓库。升级 Hunyuan3D upstream 后必须重新审查该补丁，而不是盲目套用。

## 5. 配置系统

默认配置位于 `config/default.yaml`：

```yaml
server: {host: "127.0.0.1", port: 8080}
models: {allow_download: true}
runtime: {device: "cuda", dtype: "float16"}
generation:
  default_seed: 12345
  default_format: "glb"
  default_face_count: 40000
  default_shape_steps: 50
```

所有 `paths` 必须是绝对路径。Pydantic 使用 `extra="forbid"`，拼错字段会直接失败。

环境变量覆盖格式是：

```text
LOCAL_3D_<SECTION>__<FIELD>
```

示例：

```bash
export LOCAL_3D_MODELS__ALLOW_DOWNLOAD=false
export LOCAL_3D_SERVER__PORT=8081
export LOCAL_3D_GENERATION__DEFAULT_FACE_COUNT=30000
export HUNYUAN3D_SOURCE_ROOT=/home/mcl/workspace/hunyuan3d/Hunyuan3D-2
export U2NET_HOME=/home/mcl/models/rembg
```

注意：`scripts/start_server.sh` 当前固定将 Uvicorn 绑定到 `127.0.0.1:8080`。修改 YAML 中的 host/port 不会自动改变这个脚本的命令行参数；如需变更本机端口，应同步修改启动方式和 Skill 的 `BASE_URL`。

## 6. 模型清单与解析

`config/model_manifest.yaml` 定义：

- 逻辑模型名。
- 固定本地目录。
- 官方 Hugging Face repo。
- 锁定 revision。
- 必需文件和必需目录。

模型状态：

- `READY`：所有清单要求均满足。
- `MISSING`：本地根目录不存在。
- `PARTIAL`：目录存在，但缺少核心文件、子目录或 safetensors shard。

`validate_model()` 会解析 `*.safetensors.index.json` 的 `weight_map` 并检查每个 shard。不能仅凭目录存在判断模型完整。

解析顺序：

```text
固定本地路径
  → manifest 完整性检查
  → READY 时立即使用本地路径
  → MISSING/PARTIAL 且 allow_download=true
  → snapshot_download 到固定 local_dir
  → 再次验证
  → 推理只使用解析后的本地路径
```

检查不会下载：

```bash
conda run -n hunyuan3d python scripts/check_models.py
```

显式补齐主模型：

```bash
conda run -n hunyuan3d python scripts/download_models.py --model hunyuan3d
conda run -n hunyuan3d python scripts/download_models.py --model hunyuandit
conda run -n hunyuan3d python scripts/download_models.py --all
```

项目不会覆盖 `HF_ENDPOINT`、`HTTP_PROXY` 或 `HTTPS_PROXY`。

rembg 是特殊情况：不透明输入才需要它。`u2net.onnx` 已存在时直接使用；缺失且允许下载时由 rembg 保存到 `U2NET_HOME`；缺失且禁止下载时抛出明确错误。透明 PNG 不触发 rembg。

## 7. 推理链路

### 7.1 Image-to-3D

```text
图片路径校验（≤100 MiB）
  → 复制为 job/input.png
  → 必要时 rembg
  → Hunyuan3DDiTFlowMatchingPipeline
  → raw_mesh.glb
  → 可选 Paint
  → model.glb
```

Shape 初始化参数包括：

```text
subfolder=hunyuan3d-dit-v2-0
variant=fp16
use_safetensors=true
device=cuda
dtype=float16
```

V0.1 不启用 FlashVDM/Turbo。

### 7.2 Text-to-3D

```text
prompt
  → HunyuanDiT
  → condition.png
  → 释放 HunyuanDiT、gc、empty_cache
  → Hunyuan3D Shape
  → raw_mesh.glb
  → 可选 Paint
  → model.glb
```

同一 seed 会传给 HunyuanDiT 和 Shape。GPU 算子、依赖版本和硬件变化仍可能导致非逐位一致结果。

### 7.3 Existing Mesh-to-Texture

```text
mesh + condition image
  → 格式/文件校验
  → trimesh 加载
  → FloaterRemover
  → DegenerateFaceRemover
  → FaceReducer
  → processed_mesh.glb
  → Hunyuan3DPaintPipeline
  → model.glb
  → UV/material/texture image 验证
```

默认 `face_count=40000`。提高面数会增加预处理、Paint 和输出开销。

## 8. GPU 生命周期

所有 API 生成请求共享一个 `GpuTaskLock`，一次只允许一个 GPU 工作流执行。

关键释放规则：

- Image-to-3D 不加载 HunyuanDiT。
- Text-to-3D 加载 HunyuanDiT 前释放 Shape 和 Paint。
- 生成 `condition.png` 后释放 HunyuanDiT。
- Paint 前将 raw mesh 重新加载到 CPU，释放 Shape 和 HunyuanDiT，再清理 CUDA cache。
- OOM 时释放当前 manager 中的所有 pipeline，并执行垃圾回收和 `torch.cuda.empty_cache()`。

不要在服务外长期创建第二份 Shape/Paint/HunyuanDiT pipeline。不要让测试和常驻服务同时占用同一块 GPU。

## 9. API 实现

端点：

```text
GET  /health
POST /generate/text
POST /generate/image
POST /generate/texture
```

`/health` 只检查 CUDA、模型清单和路径，不加载大型模型。

生成请求是同步请求。GPU 锁的等待时间也包含在 HTTP 请求生命周期内，客户端 timeout 必须至少为 900 秒。

Pydantic 限制：

- `seed`：0 到 `2^63-1`。
- `shape_steps`：1 到 200。
- `face_count`：100 到 1,000,000。
- `prompt`：去除首尾空格后 1 到 1000 字符。
- 图片和 mesh：必须存在、是普通文件、后缀受支持且不超过 100 MiB。
- 输出 `format`：schema 接受 `glb`、`obj`、`fbx`。

当前格式能力：

- GLB：主要格式，完整支持。
- OBJ：通过 trimesh 转换并重新验证。
- FBX：当前实现会明确报错；即使系统检测到 Blender，内存 mesh 的 FBX 转换路径也尚未实现。

异常格式：

```json
{
  "error": {
    "code": "CUDA_OUT_OF_MEMORY",
    "message": "...",
    "details": {
      "allocated_bytes": 0,
      "reserved_bytes": 0,
      "total_bytes": 0
    }
  }
}
```

CUDA OOM 返回 HTTP 503；其他推理异常映射为 `GENERATION_FAILED`。

## 10. Job 与 metadata

任务目录名：

```text
YYYYMMDD_HHMMSS_<8位短ID>
```

可能出现的文件：

```text
input.png            # Image 请求复制后的输入
condition.png        # Text 请求生成的条件图
raw_mesh.glb         # Shape 原始网格
processed_mesh.glb   # Paint 前处理结果
model.glb/model.obj  # 最终结果
metadata.json
generation.log
```

并非每种工作流都会生成全部中间文件。`generation.log` 当前只写入任务启动时间；结构化状态、耗时和错误以 `metadata.json` 为准。

metadata 记录：

- 状态、seed、prompt、shape steps、face count。
- device、dtype、UTC 开始时间、总耗时。
- 输入图、条件图、输入 mesh 和输出路径。
- 模型本地路径、revision、是否发生下载。
- generation 阶段耗时和 `torch.cuda.max_memory_allocated()` 峰值。
- 失败代码和错误信息。

metadata 使用临时文件加原子替换写入，减少进程异常时留下半截 JSON 的风险。

## 11. 网格验证

普通 GLB/OBJ 验证：

- trimesh 可重新加载。
- vertices > 0。
- faces > 0。
- 顶点坐标有限。
- bounding box 有效且至少有一个非零维度。

纹理产物还要求：

- UV 存在且非空。
- material 存在。
- material 的 `image` 或 PBR 纹理槽中存在实际图像。

Paint 返回文件但缺少实际纹理时，任务按失败处理。

## 12. 测试

### 12.1 常规测试

```bash
cd /home/mcl/workspace/3D-Assets-Agent
conda run -n hunyuan3d python -m pytest tests -q
```

GPU 测试默认跳过，因此该命令适合日常开发。当前验证基线是 `28 passed, 3 skipped`。

按层运行：

```bash
conda run -n hunyuan3d python -m pytest tests/unit -q
conda run -n hunyuan3d python -m pytest tests/integration -q
```

单元测试覆盖配置、resolver、shard、metadata、mesh、生命周期、GPU 锁和 Skill 客户端。集成测试使用 mock backend 验证 API contract，不加载大模型。

### 12.2 GPU E2E

运行前确认服务已停止，并检查空闲显存：

```bash
nvidia-smi
```

执行三条真实链路：

```bash
cd /home/mcl/workspace/3D-Assets-Agent
RUN_GPU_E2E=1 conda run -n hunyuan3d \
  python -m pytest tests/gpu/test_e2e.py -q -s --durations=3
```

这些测试会实际加载大型模型并生成临时 GLB。不得把 GPU E2E 的跳过结果当作部署通过。

### 12.3 其他验证

```bash
conda run -n hunyuan3d python -m pip check
conda run -n hunyuan3d python scripts/check_environment.py
conda run -n hunyuan3d python scripts/check_models.py
conda run -n hunyuan3d python /home/mcl/.codex/skills/.system/skill-creator/scripts/quick_validate.py skill/local-3d-agent
```

`check_models.py` 在任一清单项目不是 READY 时返回非零状态。当前 rembg 缺失时，即使 Shape/Paint/HunyuanDiT 可用，该命令也会报告失败；根据输入是否透明判断是否必须先补齐 rembg。

## 13. 调试流程

### 服务无法启动

1. 用 `conda run -n hunyuan3d python scripts/check_environment.py` 检查 Python/CUDA。
2. 检查 `HUNYUAN3D_SOURCE_ROOT`。
3. 直接运行 `scripts/start_server.sh`，保留前台异常栈。
4. 确认端口未被其他进程占用。

### 模型 MISSING/PARTIAL

1. 运行 `scripts/check_models.py`。
2. 根据输出确认是根目录、核心文件还是 shard 缺失。
3. 确认 manifest 的 revision 和目标路径。
4. 允许联网时使用 `download_models.py` 补齐主模型。
5. 不要删除整个模型目录后盲目重下。

### Paint 自定义 pipeline 失败

如果 Diffusers 提示本地 custom pipeline 需要 `trust_remote_code=True`，运行幂等的 `scripts/apply_upstream_patches.sh`，然后重新启动服务。

### CUDA OOM

1. 查看 API 返回的 allocated/reserved/total。
2. 确认没有重复启动多个服务。
3. 用 `nvidia-smi` 查看占用，但不要自动终止不属于自己的进程。
4. 降低 `face_count` 或关闭 `texture` 以缩小工作流。
5. 服务已尝试释放自身模型；必要时正常重启服务。

### 纹理 GLB 验证失败

依次检查 `processed_mesh.glb`、条件图、Paint 日志和最终 GLB 的 UV/material/texture。不要把“文件存在且非空”视为纹理成功。

## 14. 开发约定

- 使用 `src/` layout，新模块放在 `src/local_3d_agent/` 下。
- 配置和请求字段优先使用 Pydantic 验证。
- 下载决策集中在 resolver/setup 层，不散落到各 pipeline。
- GPU 模型由 `ModelManager` 持有；新增 pipeline 必须明确 acquire/release 时机。
- 新功能先增加不加载大模型的单元或集成测试。
- GPU 行为变化需要真实 GPU E2E。
- 不提交模型、生成资产、输入图片、`.env`、token 或 cache。
- 不把 Hunyuan3D upstream 直接复制到本仓库。

## 15. 提交和发布

提交前：

```bash
git diff --check
conda run -n hunyuan3d python -m pytest tests -q
conda run -n hunyuan3d python -m pip check
git status --short
```

推送前：

```bash
git log --oneline --decorate -10
git remote -v
git push origin main
```

禁止 force push 或替换远端 owner/name。部署证据更新到 `deployment_manifest.json`，至少记录项目提交、上游提交、Python、Torch、CUDA、驱动、GPU、模型路径及 E2E 结果。

项目代码使用仓库 LICENSE；Hunyuan3D upstream、Hunyuan3D 模型和 HunyuanDiT 分别遵循其上游许可证，不能被项目 LICENSE 覆盖。

## 16. 当前非目标和已知限制

V0.1 不实现：

- 多 GPU 和 GPU 并发。
- 异步 Job Queue。
- Web UI、认证或 LAN API。
- 自动 daemon 管理。
- Docker/Kubernetes。
- FlashVDM/Turbo 优化。
- 模型训练或微调。
- 可用的 FBX 转换链路。

扩展这些能力前应先更新设计规格、测试门槛和安全边界。

