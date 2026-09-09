import os
import pytest

def pytest_collection_modifyitems(items):
    if os.environ.get("RUN_GPU_E2E") == "1":
        return
    marker = pytest.mark.skip(reason="set RUN_GPU_E2E=1 to run large-model GPU tests")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(marker)

