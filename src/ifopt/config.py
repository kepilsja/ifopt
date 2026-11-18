"""TOML configuration -> frozen dataclasses (stdlib ``tomllib``)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .geometry import COEFFICIENTS, PORTS
from .materials import MaterialLibrary


@dataclass(frozen=True)
class SourceCfg:
    wavelength_nm: float
    polarization: str | float  # "s" | "p" | "unpolarized" | azimuth deg (0 = p, 90 = s)

    def pol_weights(self) -> tuple[float, float]:
        """(w_s, w_p) intensity weights; exact for isotropic stacks (s and p do not couple)."""
        import math
        p = self.polarization
        if isinstance(p, str):
            p = p.lower()
            if p == "s":
                return 1.0, 0.0
            if p == "p":
                return 0.0, 1.0
            if p in ("unpolarized", "unpolarised", "u"):
                return 0.5, 0.5
            raise ValueError(f"unknown polarization {p!r}")
        psi = math.radians(float(p))
        return math.sin(psi) ** 2, math.cos(psi) ** 2


@dataclass(frozen=True)
class PlateCfg:
    substrate: str
    substrate_thickness_mm: float
    coated_face: int                       # 1 or 2: the source whose side carries the (front) coating


@dataclass(frozen=True)
class CoatingCfg:
    materials: tuple[str, ...]             # ambient -> substrate order
    thickness_nm: tuple[float, ...] | None
    bounds_nm: dict[str, tuple[float, float]]
    optimize: bool = True                  # back face: False = fixed layers (thickness_nm required)

    def bounds(self) -> tuple[list[float], list[float]]:
        default = self.bounds_nm.get("default", (5.0, 500.0))
        lo, hi = [], []
        for m in self.materials:
            a, b = self.bounds_nm.get(m, default)
            lo.append(float(a))
            hi.append(float(b))
        return lo, hi


@dataclass(frozen=True)
class ObjectiveCfg:
    maximize: tuple[str, ...]
    weights: tuple[float, ...]
    form: str = "least_squares"            # least_squares | log | softmin
    both_polarizations: bool = False
    softmin_temperature: float = 0.02


@dataclass(frozen=True)
class OptimizerCfg:
    n_starts: int = 256
    adam_steps: int = 600
    adam_lr: float = 0.05
    top_k: int = 8
    seed: int = 0
    global_de: bool = False
    de_maxiter: int = 300
    de_popsize: int = 20


@dataclass(frozen=True)
class Config:
    path: Path
    aoi_deg: float
    ambient: str
    sources: dict[int, SourceCfg]
    plate: PlateCfg
    coating: CoatingCfg                    # front face = face of source `plate.coated_face`
    back: CoatingCfg                       # opposite face (may be empty = bare)
    objective: ObjectiveCfg
    optimizer: OptimizerCfg
    library: MaterialLibrary = field(compare=False, repr=False)

    # ---- design vector x = [front thicknesses, back thicknesses (only if the back is optimised)] ----
    @property
    def n_front(self) -> int:
        return len(self.coating.materials)

    @property
    def back_optimized(self) -> bool:
        return self.back.optimize and len(self.back.materials) > 0

    def design_materials(self) -> tuple[str, ...]:
        return self.coating.materials + (self.back.materials if self.back_optimized else ())

    def design_bounds(self) -> tuple[list[float], list[float]]:
        lo, hi = self.coating.bounds()
        if self.back_optimized:
            blo, bhi = self.back.bounds()
            lo, hi = lo + blo, hi + bhi
        return lo, hi

    def split(self, x):
        """Design vector -> (front thicknesses, back thicknesses); works on traced JAX arrays."""
        import jax.numpy as jnp
        x = jnp.asarray(x, float)
        front = x[: self.n_front]
        if self.back_optimized:
            return front, x[self.n_front:]
        return front, jnp.asarray(self.back.thickness_nm or (), float)

    def configured_design(self):
        """Design vector from thicknesses given in the config, or None if any are missing."""
        import numpy as np
        if self.coating.thickness_nm is None:
            return None
        parts = [self.coating.thickness_nm]
        if self.back_optimized:
            if self.back.thickness_nm is None:
                return None
            parts.append(self.back.thickness_nm)
        return np.concatenate([np.asarray(p, float) for p in parts])


def _pol(v):
    return v if isinstance(v, str) else float(v)


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    base = path.parent
    project_root = base.parent if base.name == "config" else base

    sysc = raw.get("system", {})
    sources = {i: SourceCfg(float(raw[f"source{i}"]["wavelength_nm"]), _pol(raw[f"source{i}"].get("polarization", "s")))
               for i in (1, 2)}

    pl = raw["plate"]
    coated = pl.get("coated_face", "source1")
    coated_face = int(str(coated).removeprefix("source"))
    if coated_face not in (1, 2):
        raise ValueError("plate.coated_face must be 'source1' or 'source2'")
    plate = PlateCfg(pl["substrate"], float(pl.get("substrate_thickness_mm", 1.0)), coated_face)

    def coating_from(sec: dict, name: str, optimize_default: bool) -> CoatingCfg:
        mats = tuple(sec.get("materials", ()))
        th = sec.get("thickness_nm")
        if th is not None:
            th = tuple(float(x) for x in th)
            if len(th) != len(mats):
                raise ValueError(f"{name}.thickness_nm must match {name}.materials in length")
        bounds = {k: (float(v[0]), float(v[1])) for k, v in sec.get("bounds_nm", {}).items()}
        return CoatingCfg(mats, th, bounds, bool(sec.get("optimize", optimize_default)))

    coating = coating_from(raw["coating"], "coating", True)
    if "back_coating" in raw:                      # optimisable (default) or fixed back-face coating
        if pl.get("back_materials"):
            raise ValueError("use either [back_coating] or plate.back_materials, not both")
        back = coating_from(raw["back_coating"], "back_coating", True)
    else:                                          # legacy: fixed layers listed under [plate]
        back = coating_from({"materials": pl.get("back_materials", ()),
                             "thickness_nm": pl.get("back_thickness_nm", ())}, "plate.back", False)
    if back.materials and not back.optimize and back.thickness_nm is None:
        raise ValueError("a fixed back coating needs thickness_nm")

    ob = raw.get("objective", {})
    if "port" in ob:
        maximize = PORTS[ob["port"]]
    else:
        maximize = tuple(ob.get("maximize", ("T1", "R2")))
    for c in maximize:
        if c not in COEFFICIENTS:
            raise ValueError(f"objective coefficient {c!r} not in {COEFFICIENTS}")
    weights = tuple(float(w) for w in ob.get("weights", [1.0] * len(maximize)))
    if len(weights) != len(maximize):
        raise ValueError("objective.weights must match the number of maximised coefficients")
    objective = ObjectiveCfg(maximize, weights, ob.get("form", "least_squares"),
                             bool(ob.get("both_polarizations", False)),
                             float(ob.get("softmin_temperature", 0.02)))

    op = raw.get("optimizer", {})
    optimizer = OptimizerCfg(**{k: v for k, v in op.items() if k in OptimizerCfg.__dataclass_fields__})

    library = MaterialLibrary.from_config_entries(raw.get("materials", {}).get("custom", []), base,
                                                  project_root / "materials")
    return Config(path, float(sysc.get("aoi_deg", 45.0)), sysc.get("ambient", "Air"), sources, plate,
                  coating, back, objective, optimizer, library)


def with_objective(cfg: Config, maximize: tuple[str, ...], weights: tuple[float, ...] | None = None) -> Config:
    from dataclasses import replace
    weights = weights or tuple(1.0 for _ in maximize)
    return replace(cfg, objective=replace(cfg.objective, maximize=tuple(maximize), weights=tuple(weights)))
