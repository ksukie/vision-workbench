"""Entry point for the lightweight Windows-only desktop executable."""

from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ["VISION_WORKBENCH_BASE_EXE"] = "1"

source_root = Path(__file__).resolve().parents[1] / "src"
if source_root.is_dir():
    sys.path.insert(0, str(source_root))

from vision_workbench.desktop.app import main


if __name__ == "__main__":
    main()
