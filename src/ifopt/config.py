"""TOML configuration (target-table schema) -> frozen dataclasses (stdlib ``tomllib``).

Layout::

    [system]       ambient
    [substrate]    material, mode (incoherent | coherent | semi_infinite | none), thickness_mm, exit_medium
    [front]        coating on the front face (ambient -> substrate order), thickness_nm, optimize, bounds_nm
    [back]         optional coating on the back face (incoherent / coherent modes only)
    [[targets]]    the design specification, see ``targets.py``
    [objective]    form (power | minimax | log), power, temperature
    [[report.sums]] optional named sums such as output ports: terms = ["T@1", "R@2"]
    [optimizer]    multi-start / polish settings
    [[materials.custom]] constant | sellmeier | csv materials
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .materials import MaterialLibrary
from .targets import TargetRow, TargetTable, parse_targets

MODES = ("incoherent", "coherent", "semi_infinite", "none")
PLATE_MODES = ("incoherent", "coherent")      # modes with a finite substrate and a back face
FORMS = ("power", "minimax", "log")


@dataclass(frozen=True)
class SubstrateCfg:
    material: str
    mode: str = "incoherent"
    thickness_mm: float = 1.0
    exit_medium: str | None = None              # mode "none": medium behind the stack (default: ambient)


@dataclass(frozen=True)
class CoatingCfg:
    materials: tuple[str, ...]                  # ambient -> substrate order
    thickness_nm: tuple[float, ...] | None
    bounds_nm: dict[str, tuple[float, float]]
    optimize: bool = True                       # False = fixed layers (thickness_nm required)

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
    form: str = "power"                         # power | minimax | log
    power: float = 2.0                          # MF = (sum w |d|^p / sum w)^(1/p)
    temperature: float = 0.01                   # minimax smoothing


@dataclass(frozen=True)
class ReportSum:
    name: str
    terms: tuple[str, ...]


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
    ambient: str
    substrate: SubstrateCfg
    front: CoatingCfg
    back: CoatingCfg
    targets: tuple[TargetRow, ...]
    objective: ObjectiveCfg
    optimizer: OptimizerCfg
    report_sums: tuple[ReportSum, ...]
    table: TargetTable = field(compare=False, repr=False)
    library: MaterialLibrary = field(compare=False, repr=False)

    @property
    def mode(self) -> str:
        return self.substrate.mode

    @property
    def has_back_face(self) -> bool:
        return self.substrate.mode in PLATE_MODES

    @property
    def exit_material(self) -> str:
        """Medium on the far side of the front coating: the substrate, or exit_medium in mode 'none'."""
        if self.substrate.mode == "none":
            return self.substrate.exit_medium or self.ambient
        return self.substrate.material

    # ---- design vector x = [front thicknesses, back thicknesses (only if the back is optimised)] ----
    @property
    def n_front(self) -> int:
        return len(self.front.materials)

    @property
    def back_optimized(self) -> bool:
        return self.back.optimize and len(self.back.materials) > 0

    def design_materials(self) -> tuple[str, ...]:
        return self.front.materials + (self.back.materials if self.back_optimized else ())

    def design_bounds(self) -> tuple[list[float], list[float]]:
        lo, hi = self.front.bounds()
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
        if self.front.thickness_nm is None:
            return None
        parts = [self.front.thickness_nm]
        if self.back_optimized:
            if self.back.thickness_nm is None:
                return None
            parts.append(self.back.thickness_nm)
        return np.concatenate([np.asarray(p, float) for p in parts])


def _coating(sec: dict, name: str) -> CoatingCfg:
    mats = tuple(sec.get("materials", ()))
    th = sec.get("thickness_nm")
    if th is not None:
        th = tuple(float(x) for x in th)
        if len(th) != len(mats):
            raise ValueError(f"[{name}] thickness_nm must match materials in length")
    bounds = {k: (float(v[0]), float(v[1])) for k, v in sec.get("bounds_nm", {}).items()}
    c = CoatingCfg(mats, th, bounds, bool(sec.get("optimize", True)))
    if c.materials and not c.optimize and c.thickness_nm is None:
        raise ValueError(f"[{name}] optimize = false needs thickness_nm")
    return c


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    legacy = [k for k in ("source1", "source2", "plate", "coating", "back_coating") if k in raw]
    if legacy:
        raise ValueError(f"{path.name}: old config format ({', '.join(legacy)}) is no longer supported; "
                         "describe the operating points as [[targets]] rows - see config/combiner_45.toml")
    base = path.parent
    project_root = base.parent if base.name == "config" else base

    su = raw.get("substrate", {})
    if "material" not in su:
        raise ValueError("[substrate] material is required")
    substrate = SubstrateCfg(su["material"], su.get("mode", "incoherent"), float(su.get("thickness_mm", 1.0)),
                             su.get("exit_medium"))
    if substrate.mode not in MODES:
        raise ValueError(f"[substrate] mode must be one of {MODES}")

    if "front" not in raw:
        raise ValueError("[front] coating section is required")
    front = _coating(raw["front"], "front")
    back = _coating(raw.get("back", {}), "back")
    if back.materials and substrate.mode not in PLATE_MODES:
        raise ValueError(f"[back] layers need a finite substrate (mode {PLATE_MODES}), not mode {substrate.mode!r}")

    targets = parse_targets(raw.get("targets", []))
    table = TargetTable.build(targets)

    ob = raw.get("objective", {})
    objective = ObjectiveCfg(ob.get("form", "power"), float(ob.get("power", 2.0)), float(ob.get("temperature", 0.01)))
    if objective.form not in FORMS:
        raise ValueError(f"[objective] form must be one of {FORMS}")
    if objective.power < 1:
        raise ValueError("[objective] power must be >= 1")

    sums = tuple(ReportSum(str(s["name"]), tuple(s["terms"])) for s in raw.get("report", {}).get("sums", []))
    for s in sums:
        for t in s.terms:
            table.parse_term(t)

    op = raw.get("optimizer", {})
    optimizer = OptimizerCfg(**{k: v for k, v in op.items() if k in OptimizerCfg.__dataclass_fields__})

    library = MaterialLibrary.from_config_entries(raw.get("materials", {}).get("custom", []), base,
                                                  project_root / "materials")
    return Config(path, raw.get("system", {}).get("ambient", "Air"), substrate, front, back, targets, objective,
                  optimizer, sums, table, library)
