import os
import sys
from pathlib import Path

# Ensure repo root on path so `src` is importable when running pytest from root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Clear env between tests
def pytest_runtest_setup(item):
    for k in list(os.environ.keys()):
        if k in {
            "MODEL_ID","PRECISION","MAX_NEW_TOKENS","TEMPERATURE",
            "TOP_P","DEVICE_MAP","SYSTEM_PROMPT"
        }:
            os.environ.pop(k, None)
