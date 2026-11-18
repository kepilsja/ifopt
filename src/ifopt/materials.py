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
            else:
                raise ValueError(f"unknown material type {typ!r} for {name}")
        return cls(custom, materials_dir)

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
