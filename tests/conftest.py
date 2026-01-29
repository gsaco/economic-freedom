import sys
from pathlib import Path

# Ensure src/ is on path for cleanroom modules
root = Path(__file__).resolve().parents[1]
src = root / "src"
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
