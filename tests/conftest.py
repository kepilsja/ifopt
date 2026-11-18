import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import ifopt  # noqa: E402,F401  (x64)


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def cfg():
    from ifopt.config import load_config
    return load_config(ROOT / "config" / "example.toml")


@pytest.fixture(scope="session")
def lib():
    from ifopt.materials import MaterialLibrary
    return MaterialLibrary()
