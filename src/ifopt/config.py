"""Specification objects: the filter system and its parts (code or TOML, same validation).

Build in code::

    system = FilterSystem(
        substrate=Substrate("FusedSilica", mode="semi_infinite"),
        front=Coating(alternating("TiO2", "FusedSilica", 21), bounds_nm={"default": (5, 400)}),
        targets=[Beam("pass", wavelength_nm=(520, 545), points=26).min("T", 0.98)],
    )

or load / save the TOML layout::

    [system]       ambient
    [substrate]    material, mode (incoherent | coherent | semi_infinite | none), thickness_mm, exit_medium
    [front]        coating on the front face (ambient -> substrate order), thickness_nm, optimize, bounds_nm
    [back]         optional coating on the back face (incoherent / coherent modes only)
    [[targets]]    the design specification, see ``targets.py``
    [objective]    form (power | minimax | log), power, temperature
    [[report.sums]] optional named sums such as output ports: terms = ["T@1", "R@2"]
    [optimizer]    multi-start / polish settings
    [[materials.custom]] constant | sellmeier | table | csv materials

All objects are frozen; vary them with ``dataclasses.replace`` or ``FilterSystem.replace`` (which
re-validates).  Materials may be given as ``Material`` objects; they are registered in the system's
``MaterialLibrary`` and replaced by their names.
"""
from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from .materials import Material, MaterialLibrary
from .targets import Target, TargetTable, parse_targets, resolve_targets

MODES = ("incoherent", "coherent", "semi_infinite", "none")
PLATE_MODES = ("incoherent", "coherent")      # modes with a finite substrate and a back face
FORMS = ("power", "minimax", "log")


@dataclass(frozen=True)
class Substrate:
    material: str | Material
    mode: str = "incoherent"
    thickness_mm: float = 1.0
    exit_medium: str | Material | None = None   # mode "none": medium behind the stack (default: ambient)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"substrate mode must be one of {MODES}, got {self.mode!r}")
        object.__setattr__(self, "thickness_mm", float(self.thickness_mm))
        if self.thickness_mm <= 0:
            raise ValueError("substrate thickness_mm must be > 0")

    def to_dict(self) -> dict:
        d = {"material": _name(self.material), "mode": self.mode, "thickness_mm": self.thickness_mm}
        if self.exit_medium is not None:
            d["exit_medium"] = _name(self.exit_medium)
        return d


@dataclass(frozen=True)
class Coating:
    materials: tuple = ()                       # ambient -> substrate order; names or Material objects
    thickness_nm: tuple[float, ...] | None = None
    bounds_nm: dict | None = None               # {"default": (lo, hi), "<material>": (lo, hi)}
    optimize: bool = True                       # False = fixed layers (thickness_nm required)

    def __post_init__(self):
        s = object.__setattr__
        s(self, "materials", tuple(self.materials))
        if self.thickness_nm is not None:
            s(self, "thickness_nm", tuple(float(x) for x in self.thickness_nm))
            if len(self.thickness_nm) != len(self.materials):
                raise ValueError("coating thickness_nm must match materials in length")
            if any(x < 0 for x in self.thickness_nm):
                raise ValueError("coating thicknesses must be >= 0")
        bounds = {}
        for k, v in (self.bounds_nm or {}).items():
            lo, hi = float(v[0]), float(v[1])
            if not 0 <= lo <= hi:
                raise ValueError(f"coating bounds_nm[{k!r}] must satisfy 0 <= lo <= hi")
            bounds[str(k)] = (lo, hi)
        s(self, "bounds_nm", bounds)
        s(self, "optimize", bool(self.optimize))
        if self.materials and not self.optimize and self.thickness_nm is None:
            raise ValueError("a coating with optimize = False needs thickness_nm")

    def bounds(self) -> tuple[list[float], list[float]]:
        default = self.bounds_nm.get("default", (5.0, 500.0))
        lo, hi = [], []
        for m in self.materials:
            a, b = self.bounds_nm.get(_name(m), default)
            lo.append(float(a))
            hi.append(float(b))
        return lo, hi

    def to_dict(self) -> dict:
        d = {"materials": [_name(m) for m in self.materials], "optimize": self.optimize}
        if self.thickness_nm is not None:
            d["thickness_nm"] = list(self.thickness_nm)
        d["bounds_nm"] = {k: list(v) for k, v in self.bounds_nm.items()}
        return d


@dataclass(frozen=True)
class Objective:
    form: str = "power"                         # power | minimax | log
    power: float = 2.0                          # MF = (sum w |d|^p / sum w)^(1/p)
    temperature: float = 0.01                   # minimax smoothing

    def __post_init__(self):
        if self.form not in FORMS:
            raise ValueError(f"objective form must be one of {FORMS}")
        object.__setattr__(self, "power", float(self.power))
        object.__setattr__(self, "temperature", float(self.temperature))
        if self.power < 1 or self.temperature <= 0:
            raise ValueError("objective power must be >= 1 and temperature > 0")


@dataclass(frozen=True)
class OptimizerSettings:
    n_starts: int = 256
    adam_steps: int = 600
    adam_lr: float = 0.05
    top_k: int = 8
    seed: int = 0
    global_de: bool = False
    de_maxiter: int = 300
    de_popsize: int = 20

    def __post_init__(self):
        for f in ("n_starts", "adam_steps", "top_k", "de_maxiter", "de_popsize"):
            if int(getattr(self, f)) < 1:
                raise ValueError(f"optimizer {f} must be >= 1")
            object.__setattr__(self, f, int(getattr(self, f)))
        if self.adam_lr <= 0:
            raise ValueError("optimizer adam_lr must be > 0")


@dataclass(frozen=True)
class ReportSum:
    name: str
    terms: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "terms", tuple(str(t) for t in self.terms))


def _name(m) -> str:
    return m.name if isinstance(m, Material) else str(m)


@dataclass(frozen=True)
class FilterSystem:
    substrate: Substrate
    front: Coating
    targets: tuple[Target, ...]
    back: Coating | None = None
    ambient: str | Material = "Air"
    objective: Objective = field(default_factory=Objective)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    sums: tuple[ReportSum, ...] = ()
    materials: MaterialLibrary | None = field(default=None, compare=False, repr=False)
    source: Path | None = field(default=None, compare=False, repr=False)
    table: TargetTable = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        s = object.__setattr__
        lib = self.materials if self.materials is not None else MaterialLibrary()
        s(self, "materials", lib)

        def reg(m):
            if isinstance(m, Material):
                lib.add(m)
                return m.name
            return m

        s(self, "ambient", reg(self.ambient))
        s(self, "substrate", replace(self.substrate, material=reg(self.substrate.material),
                                     exit_medium=None if self.substrate.exit_medium is None else reg(self.substrate.exit_medium)))
        back = self.back if self.back is not None else Coating()
        s(self, "front", replace(self.front, materials=tuple(reg(m) for m in self.front.materials)))
        s(self, "back", replace(back, materials=tuple(reg(m) for m in back.materials)))
        if self.back.materials and self.substrate.mode not in PLATE_MODES:
            raise ValueError(f"back-face layers need a finite substrate (mode {PLATE_MODES}), "
                             f"not mode {self.substrate.mode!r}")
        if not self.front.materials:
            raise ValueError("the front coating needs at least one layer")
        s(self, "targets", resolve_targets(list(self.targets)))
        s(self, "sums", tuple(x if isinstance(x, ReportSum) else ReportSum(*x) for x in self.sums))
        table = TargetTable.build(self.targets)
        s(self, "table", table)
        for sm in self.sums:
            for t in sm.terms:
                table.parse_term(t)
        if self.source is not None:
            s(self, "source", Path(self.source))
        # every material must exist and cover every target wavelength (fail early, not inside JAX)
        names = {self.ambient, self.exit_material, *self.front.materials, *self.back.materials}
        if self.has_back_face:
            names.add(self.substrate.material)
        for n in names:
            lib.nk(n, table.point_lam)

    # ---- compatibility names ----
    @property
    def library(self) -> MaterialLibrary:
        return self.materials

    @property
    def report_sums(self) -> tuple[ReportSum, ...]:
        return self.sums

    @property
    def path(self) -> Path | None:
        return self.source

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

    # ---- variation ----
    def replace(self, **changes) -> "FilterSystem":
        """Copy with changes (re-validated); the material library is shared."""
        return replace(self, **changes)

    def add_targets(self, *targets: Target) -> "FilterSystem":
        return replace(self, targets=tuple(self.targets) + tuple(targets))

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
        """Design vector from thicknesses given in the spec, or None if any are missing."""
        import numpy as np
        if self.front.thickness_nm is None:
            return None
        parts = [self.front.thickness_nm]
        if self.back_optimized:
            if self.back.thickness_nm is None:
                return None
            parts.append(self.back.thickness_nm)
        return np.concatenate([np.asarray(p, float) for p in parts])

    # ---- serialisation ----
    def to_dict(self, base_dir: Path | None = None) -> dict:
        d = {"system": {"ambient": self.ambient}, "substrate": self.substrate.to_dict(), "front": self.front.to_dict()}
        if self.back.materials:
            d["back"] = self.back.to_dict()
        d["targets"] = [t.to_dict() for t in self.targets]
        d["objective"] = asdict(self.objective)
        if self.sums:
            d["report"] = {"sums": [{"name": s.name, "terms": list(s.terms)} for s in self.sums]}
        d["optimizer"] = asdict(self.optimizer)
        entries = self.materials.to_entries(base_dir)
        if entries:
            d["materials"] = {"custom": entries}
        return d

    def to_toml(self, path: str | Path | None = None) -> str:
        """TOML text; also written to ``path`` if given (CSV material references made relative to it)."""
        from .io import dumps_toml
        base = Path(path).resolve().parent if path is not None else None
        text = "# written by ifopt\n" + dumps_toml(self.to_dict(base))
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_dict(cls, raw: dict, base_dir: str | Path | None = None, materials_dir: str | Path | None = None,
                  source: str | Path | None = None, materials: MaterialLibrary | None = None) -> "FilterSystem":
        legacy = [k for k in ("source1", "source2", "plate", "coating", "back_coating") if k in raw]
        if legacy:
            raise ValueError(f"old config format ({', '.join(legacy)}) is no longer supported; "
                             "describe the operating points as [[targets]] rows - see config/combiner_45.toml")
        base = Path(base_dir) if base_dir is not None else Path.cwd()
        su = raw.get("substrate", {})
        if "material" not in su:
            raise ValueError("[substrate] material is required")
        if "front" not in raw:
            raise ValueError("[front] coating section is required")
        lib = materials if materials is not None else MaterialLibrary(
            materials_dir=Path(materials_dir) if materials_dir is not None else None)
        for m in MaterialLibrary.from_config_entries(raw.get("materials", {}).get("custom", []), base).custom.values():
            lib.add(m)

        def coating(sec: dict) -> Coating:
            return Coating(tuple(sec.get("materials", ())), sec.get("thickness_nm"), sec.get("bounds_nm"),
                           bool(sec.get("optimize", True)))

        known = {f.name for f in fields(OptimizerSettings)}
        return cls(
            substrate=Substrate(su["material"], su.get("mode", "incoherent"), su.get("thickness_mm", 1.0),
                                su.get("exit_medium")),
            front=coating(raw["front"]),
            back=coating(raw["back"]) if "back" in raw else None,
            targets=parse_targets(raw.get("targets", [])),
            ambient=raw.get("system", {}).get("ambient", "Air"),
            objective=Objective(**raw.get("objective", {})),
            optimizer=OptimizerSettings(**{k: v for k, v in raw.get("optimizer", {}).items() if k in known}),
            sums=tuple(ReportSum(str(s["name"]), tuple(s["terms"])) for s in raw.get("report", {}).get("sums", [])),
            materials=lib, source=source)

    @classmethod
    def from_toml(cls, path: str | Path, materials: MaterialLibrary | None = None) -> "FilterSystem":
        path = Path(path).resolve()
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        project_root = path.parent.parent if path.parent.name == "config" else path.parent
        try:
            return cls.from_dict(raw, base_dir=path.parent, materials_dir=project_root / "materials", source=path,
                                 materials=materials)
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from None


# backwards-compatible names
Config = FilterSystem
SubstrateCfg, CoatingCfg, ObjectiveCfg, OptimizerCfg = Substrate, Coating, Objective, OptimizerSettings


def load_config(path: str | Path) -> FilterSystem:
    return FilterSystem.from_toml(path)
