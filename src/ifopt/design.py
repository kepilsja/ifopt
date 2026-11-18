"""A design: layer thicknesses of the front coating and, if present, the back coating (nm)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import FilterSystem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Design:
    front_nm: tuple[float, ...]
    back_nm: tuple[float, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "front_nm", tuple(float(v) for v in np.ravel(self.front_nm)))
        object.__setattr__(self, "back_nm", tuple(float(v) for v in np.ravel(self.back_nm)))
        if any(v < 0 or not np.isfinite(v) for v in self.front_nm + self.back_nm):
            raise ValueError("layer thicknesses must be finite and >= 0")

    # ---- construction ----
    @classmethod
    def from_x(cls, system: FilterSystem, x) -> "Design":
        x = np.asarray(x, float).ravel()
        if x.size != len(system.design_materials()):
            raise ValueError(f"design vector has {x.size} entries, the system needs {len(system.design_materials())}")
        front = x[: system.n_front]
        back = x[system.n_front:] if system.back_optimized else np.asarray(system.back.thickness_nm or (), float)
        return cls(front, back)

    @classmethod
    def quarter_wave(cls, system: FilterSystem) -> "Design":
        """Quarter-wave seed (see ``objectives.quarter_wave_seed``)."""
        from .objectives import quarter_wave_seed
        return cls.from_x(system, quarter_wave_seed(system))

    @classmethod
    def from_system(cls, system: FilterSystem) -> "Design":
        """Thicknesses written in the specification ([front] / [back] thickness_nm)."""
        x = system.configured_design()
        if x is None:
            raise ValueError("the system does not specify all layer thicknesses")
        return cls.from_x(system, x)

    # ---- use ----
    def x(self, system: FilterSystem) -> np.ndarray:
        """Optimisation vector: front, then back only when the back coating is optimised."""
        self.check(system, warn_bounds=False)
        parts = [self.front_nm] + ([self.back_nm] if system.back_optimized else [])
        return np.concatenate([np.asarray(p, float) for p in parts])

    def check(self, system: FilterSystem, warn_bounds: bool = True) -> "Design":
        if len(self.front_nm) != system.n_front:
            raise ValueError(f"design has {len(self.front_nm)} front layers, the system has {system.n_front}")
        if len(self.back_nm) != len(system.back.materials):
            raise ValueError(f"design has {len(self.back_nm)} back layers, the system has {len(system.back.materials)}")
        if warn_bounds:
            lo, hi = map(np.asarray, system.design_bounds())
            x = np.concatenate([self.front_nm, self.back_nm if system.back_optimized else ()])
            out = np.flatnonzero((x < lo - 1e-9) | (x > hi + 1e-9))
            if out.size:
                logger.warning("design layers %s are outside their bounds", out.tolist())
        return self

    def perturbed(self, rel_sigma: float, rng: np.random.Generator | None = None) -> "Design":
        rng = rng or np.random.default_rng()
        f = lambda v: np.clip(np.asarray(v) * (1 + rel_sigma * rng.standard_normal(len(v))), 0, None)
        return Design(f(self.front_nm), f(self.back_nm))

    @property
    def total_nm(self) -> float:
        return float(sum(self.front_nm) + sum(self.back_nm))

    # ---- files (design.json layout used by the CLI) ----
    def to_dict(self, system: FilterSystem | None = None) -> dict:
        d = {"thickness_nm": list(self.front_nm), "back_thickness_nm": list(self.back_nm)}
        if system is not None:
            d = {"materials": list(system.front.materials), **d, "back_materials": list(system.back.materials),
                 "back_optimized": system.back_optimized}
        return d

    def save(self, path: str | Path, system: FilterSystem | None = None, **extra) -> Path:
        path = Path(path)
        path.write_text(json.dumps({**(self.to_dict(system)), **extra}, indent=2))
        logger.info("saved %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path, system: FilterSystem | None = None) -> "Design":
        d = json.loads(Path(path).read_text())
        design = cls(d["thickness_nm"], d.get("back_thickness_nm", ()))
        if system is not None:
            if "materials" in d and list(d["materials"]) != list(system.front.materials):
                raise ValueError(f"{path}: front materials do not match the system")
            if d.get("back_materials") and list(d["back_materials"]) != list(system.back.materials):
                raise ValueError(f"{path}: back materials do not match the system")
            if not system.back_optimized and system.back.thickness_nm is not None:
                design = cls(design.front_nm, system.back.thickness_nm)
            design.check(system)
        return design


def as_x(system: FilterSystem, design) -> np.ndarray:
    """Design object or plain vector -> optimisation vector."""
    if isinstance(design, Design):
        return design.x(system)
    x = np.asarray(design, float).ravel()
    if x.size != len(system.design_materials()):
        raise ValueError(f"design vector has {x.size} entries, the system needs {len(system.design_materials())}")
    return x
