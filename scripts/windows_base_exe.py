"""Entry point for the lightweight Windows-only desktop executable."""

from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ["VISION_WORKBENCH_BASE_EXE"] = "1"

source_root = Path(__file__).resolve().parents[1] / "src"
if source_root.is_dir():
    sys.path.insert(0, str(source_root))

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "--vision-workbench-self-test":
        from vision_workbench.self_test import main as self_test_main

        raise SystemExit(self_test_main(["--expected-mode", "single-file", *sys.argv[2:]]))
    if command == "--vision-workbench-apply-update":
        from vision_workbench.update_helper import main as update_main

        raise SystemExit(update_main(sys.argv[2:]))

    from vision_workbench.desktop.app import main

    main()
