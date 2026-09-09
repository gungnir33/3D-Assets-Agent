import threading

class GpuTaskLock:
    def __init__(self):
        self._lock = threading.Lock()
    def __enter__(self):
        self._lock.acquire(); return self
    def __exit__(self, exc_type, exc, tb):
        self._lock.release()

