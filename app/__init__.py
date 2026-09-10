"""RetentionLab model serving layer (internal package name: churnguard).

The analytical package lives in ``src/churnguard`` and is normally imported via
an editable install or the pytest ``pythonpath`` setting. Some deployment
runtimes copy the repository and import ``app`` directly without installing the
project, so ``src`` is added to ``sys.path`` here if it is not already
importable. The insert is idempotent and a no-op in an installed environment.
"""

import sys
from importlib.util import find_spec
from pathlib import Path

if find_spec("churnguard") is None:  # pragma: no cover - environment dependent
    _src = Path(__file__).resolve().parents[1] / "src"
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
