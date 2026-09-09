from pathlib import Path

class TextToImagePipeline:
    def __init__(self, backend):
        self.backend = backend

    def generate(self, prompt: str, output: Path, *, seed: int) -> Path:
        image = self.backend(prompt, seed=seed)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        return output

