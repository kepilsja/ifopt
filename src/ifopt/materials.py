"""Material library: complex refractive index N = n + i k (k >= 0) as a function of wavelength.

Sources, in lookup order:
  1. materials defined in the config file (constant, Sellmeier or CSV table),
  2. CSV files dropped into ``materials/`` (columns: wavelength_nm, n, k; '#' comments allowed),
  3. built-in Sellmeier glasses (FusedSilica, N-BK7) and Vacuum,
  4. the tmmax nk database (read-only; files are tabulated in micrometres).

Unlike tmmax (``jnp.interp`` silently clamps to the end of the table), every lookup outside
the tabulated range raises ``MaterialRangeError``.  Example: tmmax's ``Si`` table only covers
2-20 um, so tmmax evaluates Si at 532 nm with the 2 um value (n=3.47, k~0) without warning.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class MaterialRangeError(ValueError):
    pass


@dataclass(frozen=True)
class Material:
    name: str
    source: str
    wl_min_nm: float = 0.0
    wl_max_nm: float = float("inf")

    def nk(self, wl_nm) -> np.ndarray:
        wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
        tol = 1e-9 * max(1.0, self.wl_max_nm if np.isfinite(self.wl_max_nm) else 1.0)
        if np.any(wl < self.wl_min_nm - tol) or np.any(wl > self.wl_max_nm + tol):
            raise MaterialRangeError(
                f"{self.name} ({self.source}): wavelength(s) {wl[(wl < self.wl_min_nm) | (wl > self.wl_max_nm)]} nm "
                f"outside tabulated range [{self.wl_min_nm:g}, {self.wl_max_nm:g}] nm")
        return self._nk(wl)

    def _nk(self, wl: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantMaterial(Material):
    n: float = 1.0
    k: float = 0.0

    def _nk(self, wl):
        return np.full(wl.shape, self.n + 1j * self.k, dtype=complex)


@dataclass(frozen=True)
class SellmeierMaterial(Material):
    """n^2 = 1 + sum_i B_i lambda^2 / (lambda^2 - C_i), lambda in um, C in um^2; k = 0."""
    B: tuple = ()
    C: tuple = ()

    def _nk(self, wl):
        lam2 = (wl * 1e-3) ** 2
        n2 = 1.0 + sum(b * lam2 / (lam2 - c) for b, c in zip(self.B, self.C))
        return np.sqrt(n2).astype(complex)


@dataclass(frozen=True)
class TabulatedMaterial(Material):
    wl_nm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_tab: np.ndarray = field(default_factory=lambda: np.zeros(0))
    k_tab: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def from_arrays(cls, name, source, wl_nm, n, k):
        order = np.argsort(wl_nm)
        wl_nm, n, k = (np.asarray(a, dtype=float)[order] for a in (wl_nm, n, k))
        if np.any(k < 0):
            raise ValueError(f"{name}: negative k found; this library uses N = n + ik with k >= 0")
        return cls(name=name, source=source, wl_min_nm=float(wl_nm[0]), wl_max_nm=float(wl_nm[-1]),
                   wl_nm=wl_nm, n_tab=n, k_tab=k)

    def _nk(self, wl):
        return np.interp(wl, self.wl_nm, self.n_tab) + 1j * np.interp(wl, self.wl_nm, self.k_tab)


# Malitson (1965) fused silica, valid 0.21-6.7 um; Schott N-BK7 datasheet, valid 0.3-2.5 um.
BUILTIN: dict[str, Material] = {
    "Vacuum": ConstantMaterial(name="Vacuum", source="builtin", n=1.0, k=0.0),
    "FusedSilica": SellmeierMaterial(
        name="FusedSilica", source="builtin Sellmeier (Malitson 1965)", wl_min_nm=210, wl_max_nm=6700,
        B=(0.6961663, 0.4079426, 0.8974794), C=(0.0684043 ** 2, 0.1162414 ** 2, 9.896161 ** 2)),
    "N-BK7": SellmeierMaterial(
        name="N-BK7", source="builtin Sellmeier (Schott)", wl_min_nm=300, wl_max_nm=2500,
        B=(1.03961212, 0.231792344, 1.01046945), C=(0.00600069867, 0.0200179144, 103.560653)),
}


def tmmax_db_dir() -> Path | None:
    try:
        import tmmax  # noqa: WPS433
    except ImportError:
        return None
    return Path(os.path.dirname(tmmax.__file__)) / "nk_data" / "numpy"


def load_csv(path: Path, name: str | None = None) -> TabulatedMaterial:
    data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    if data.shape[1] != 3:
        raise ValueError(f"{path}: expected 3 columns wavelength_nm,n,k")
    return TabulatedMaterial.from_arrays(name or path.stem, f"csv:{path}", data[:, 0], data[:, 1], data[:, 2])


def same_material(a: Material, b: Material) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, TabulatedMaterial):
        return all(np.array_equal(getattr(a, f), getattr(b, f)) for f in ("wl_nm", "n_tab", "k_tab"))
    return a == b


class MaterialLibrary:
    def __init__(self, custom: dict[str, Material] | None = None, materials_dir: Path | None = None):
        self.custom = dict(custom or {})
        self.materials_dir = materials_dir
        self._cache: dict[str, Material] = {}

    @classmethod
    def from_config_entries(cls, entries: list[dict], base_dir: Path, materials_dir: Path | None = None):
        custom: dict[str, Material] = {}
        for e in entries:
            name, typ = e["name"], e["type"]
            if typ == "constant":
                custom[name] = ConstantMaterial(name=name, source="config constant",
                                                n=float(e["n"]), k=float(e.get("k", 0.0)))
            elif typ == "sellmeier":
                custom[name] = SellmeierMaterial(name=name, source="config Sellmeier",
                                                 wl_min_nm=float(e.get("wl_min_nm", 0)),
                                                 wl_max_nm=float(e.get("wl_max_nm", float("inf"))),
                                                 B=tuple(e["B"]), C=tuple(e["C"]))
            elif typ == "csv":
                custom[name] = load_csv((base_dir / e["path"]).resolve(), name)
            elif typ == "table":
                custom[name] = TabulatedMaterial.from_arrays(name, "table", e["wavelength_nm"], e["n"], e.get("k", [0.0] * len(e["n"])))
            else:
                raise ValueError(f"unknown material type {typ!r} for {name}")
        return cls(custom, materials_dir)

    # ---- registering materials from code ----------------------------------------------------
    def add(self, material: Material) -> Material:
        """Register a material under its name.  Re-adding an identical material is a no-op; a
        different material under an already registered custom name is an error."""
        old = self.custom.get(material.name)
        if old is not None and not same_material(old, material):
            raise ValueError(f"a different material named {material.name!r} is already registered")
        self.custom[material.name] = material
        self._cache.pop(material.name, None)
        return material

    def add_constant(self, name: str, n: float, k: float = 0.0) -> Material:
        return self.add(ConstantMaterial(name=name, source="constant", n=float(n), k=float(k)))

    def add_sellmeier(self, name: str, B, C, wl_min_nm: float = 0.0, wl_max_nm: float = float("inf")) -> Material:
        """n^2 = 1 + sum B_i lambda^2 / (lambda^2 - C_i) with lambda in um and C in um^2."""
        return self.add(SellmeierMaterial(name=name, source="Sellmeier", wl_min_nm=float(wl_min_nm),
                                          wl_max_nm=float(wl_max_nm), B=tuple(map(float, B)), C=tuple(map(float, C))))

    def add_table(self, name: str, wavelength_nm, n, k=None) -> Material:
        k = np.zeros(len(n)) if k is None else k
        return self.add(TabulatedMaterial.from_arrays(name, "table", wavelength_nm, n, k))

    def add_csv(self, path, name: str | None = None) -> Material:
        return self.add(load_csv(Path(path).resolve(), name))

    def to_entries(self, base_dir: Path | None = None) -> list[dict]:
        """Custom materials as ``[[materials.custom]]`` entries (CSV references kept, tables inline)."""
        out = []
        for name, m in self.custom.items():
            if isinstance(m, ConstantMaterial):
                out.append({"name": name, "type": "constant", "n": m.n, "k": m.k})
            elif isinstance(m, SellmeierMaterial):
                out.append({"name": name, "type": "sellmeier", "B": list(m.B), "C": list(m.C),
                            "wl_min_nm": m.wl_min_nm, "wl_max_nm": m.wl_max_nm})
            elif isinstance(m, TabulatedMaterial) and m.source.startswith("csv:"):
                path = Path(m.source.removeprefix("csv:"))
                ref = os.path.relpath(path, base_dir) if base_dir is not None else str(path)
                out.append({"name": name, "type": "csv", "path": ref})
            elif isinstance(m, TabulatedMaterial):
                out.append({"name": name, "type": "table", "wavelength_nm": m.wl_nm.tolist(),
                            "n": m.n_tab.tolist(), "k": m.k_tab.tolist()})
            else:
                raise TypeError(f"cannot serialise material {name!r} of type {type(m).__name__}")
        return out

    def get(self, name: str) -> Material:
        if name in self._cache:
            return self._cache[name]
        mat = self._lookup(name)
        self._cache[name] = mat
        return mat

    def _lookup(self, name: str) -> Material:
        if name in self.custom:
            return self.custom[name]
        if self.materials_dir is not None and (self.materials_dir / f"{name}.csv").exists():
            return load_csv(self.materials_dir / f"{name}.csv", name)
        if name in BUILTIN:
            return BUILTIN[name]
        key = name.removeprefix("tmmax:")
        db = tmmax_db_dir()
        if db is not None and (db / f"{key}.npy").exists():
            wl_um, n, k = np.load(db / f"{key}.npy")
            return TabulatedMaterial.from_arrays(name, "tmmax nk database", wl_um * 1e3, n, k)
        raise KeyError(f"unknown material {name!r}; known: {sorted(self.available())}")

    def available(self) -> list[str]:
        names = set(self.custom) | set(BUILTIN)
        if self.materials_dir is not None and self.materials_dir.exists():
            names |= {p.stem for p in self.materials_dir.glob("*.csv")}
        db = tmmax_db_dir()
        if db is not None:
            names |= {p.stem for p in db.glob("*.npy")}
        return sorted(names)

    def nk(self, name: str, wl_nm) -> np.ndarray:
        return self.get(name).nk(wl_nm)

    def nk_table(self, names: list[str], wl_nm) -> np.ndarray:
        """Complex index table of shape (len(names), len(wl_nm))."""
        wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
        if not names:
            return np.zeros((0, wl.size), dtype=complex)
        return np.stack([self.nk(m, wl) for m in names])
