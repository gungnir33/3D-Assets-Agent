def preprocess_mesh(mesh, face_count: int, *, floater=None, degenerate=None, reducer=None):
    if face_count <= 0:
        raise ValueError("face_count must be positive")
    if floater is None or degenerate is None or reducer is None:
        from hy3dgen.shapegen import FloaterRemover, DegenerateFaceRemover, FaceReducer
        floater = floater or FloaterRemover()
        degenerate = degenerate or DegenerateFaceRemover()
        reducer = reducer or FaceReducer()
    mesh = floater(mesh)
    mesh = degenerate(mesh)
    return reducer(mesh, max_facenum=face_count)

