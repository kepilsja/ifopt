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
    return load_config(ROOT / "config" / "combiner_45.toml")


@pytest.fixture(scope="session")
def cfg_ar():
    from ifopt.config import load_config
    return load_config(ROOT / "config" / "combiner_45_back_ar.toml")


@pytest.fixture(scope="session")
def lib():
    from ifopt.materials import MaterialLibrary
    return MaterialLibrary()
