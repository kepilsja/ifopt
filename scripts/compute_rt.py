"""Shim kept for backwards compatibility: same as `ifopt evaluate ...` (see `ifopt evaluate --help`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ifopt.cli import main  # noqa: E402

raise SystemExit(main(["evaluate", *sys.argv[1:]]))
