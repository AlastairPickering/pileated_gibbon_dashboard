import importlib
from pathlib import Path
import sys

MODULES = [
    "scripts.preprocessing",
    "scripts.prepare_occupancy",
    "scripts.pipeline",
]

missing = []
for mod in MODULES:
    try:
        importlib.import_module(mod)
    except Exception as e:
        missing.append((mod, repr(e)))

if missing:
    problems = "\n".join(f"- {m}: {e}" for m, e in missing)
    raise SystemExit(f"Smoke test import failures:\n{problems}")

# Sanity-check expected top-level folders exist
expected_dirs = ["audio", "results", "models"]
missing_dirs = [p for p in expected_dirs if not Path(p).exists()]
if missing_dirs:
    print(f"Note: missing expected folders (may be fine for CI): {missing_dirs}")

print("Smoke test passed on", sys.version)
