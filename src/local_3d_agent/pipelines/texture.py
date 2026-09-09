from pathlib import Path
from local_3d_agent.mesh.preprocess import preprocess_mesh
from local_3d_agent.mesh.validate import validate_mesh

class TexturePipeline:
    def __init__(self, backend, *, preprocessor=preprocess_mesh, validator=validate_mesh):
        self.backend = backend
        self.preprocessor = preprocessor
        self.validator = validator

    def generate(self, mesh, image: Path, job_dir: Path, *, face_count: int) -> Path:
        job_dir.mkdir(parents=True, exist_ok=True)
        processed = self.preprocessor(mesh, face_count)
        processed_path = job_dir / "processed_mesh.glb"
        processed.export(processed_path)
        textured = self.backend(processed, image=str(image))
        output = job_dir / "model.glb"
        textured.export(output)
        self.validator(output, require_texture=True)
        return output
