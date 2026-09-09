from pathlib import Path
from typing import Callable
from PIL import Image

class ImageTo3DPipeline:
    def __init__(self, backend, generator_factory: Callable[[int], object] | None = None):
        self.backend = backend
        self.generator_factory = generator_factory or self._torch_generator

    @staticmethod
    def _torch_generator(seed: int):
        import torch
        return torch.Generator(device="cuda").manual_seed(seed)

    def generate(self, image: Path, output: Path, *, seed: int, shape_steps: int) -> Path:
        source = Image.open(image).convert("RGBA")
        meshes = self.backend(image=source, num_inference_steps=shape_steps,
                              generator=self.generator_factory(seed))
        mesh = meshes[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(output)
        return output

