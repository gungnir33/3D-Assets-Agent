class ApiError(RuntimeError):
    def __init__(self, code: str, message: str, *, details=None, status_code: int = 500):
        super().__init__(message)
        self.code, self.details, self.status_code = code, details or {}, status_code

