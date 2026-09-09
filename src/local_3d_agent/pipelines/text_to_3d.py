from pathlib import Path

class TextTo3DPipeline:
    def __init__(self, *, t2i, shape, model_manager):
        self.t2i = t2i
        self.shape = shape
        self.model_manager = model_manager

    def generate(self, prompt: str, job_dir: Path, *, seed: int, shape_steps: int) -> Path:
        job_dir.mkdir(parents=True, exist_ok=True)
        condition = self.t2i.generate(prompt, job_dir / "condition.png", seed=seed)
        self.model_manager.release("t2i")
        self.model_manager.cleanup_cuda()
        return self.shape.generate(condition, job_dir / "raw_mesh.glb", seed=seed, shape_steps=shape_steps)

