"""Legacy backend tests require the products extra; base CI stays lightweight."""

import importlib.util
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("torch", "pandas", "numpy", "PIL")):
    collect_ignore = [p.name for p in Path(__file__).parent.glob("test_*.py")]
