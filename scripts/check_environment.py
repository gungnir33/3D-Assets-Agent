#!/usr/bin/env python3
import json, platform, subprocess
from pathlib import Path

def collect_environment():
    import torch
    cuda = torch.cuda.is_available()
    return {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cuda": cuda, "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "upstream": str(Path("/home/mcl/workspace/hunyuan3d/Hunyuan3D-2"))}

if __name__ == "__main__":
    data = collect_environment(); print(json.dumps(data, indent=2))
    raise SystemExit(not (data["python"].startswith("3.10.") and data["cuda"]))
