"""Target table: the design specification, as in thin-film design software.

Each target is an operating condition plus a requirement::

    (wavelength or range, AOI or range, polarisation, side) -> quantity R|T|A, kind eq|min|max, value, tolerance, weight

Targets are expanded over their wavelength x AOI grid (and s/p for ``polarization = "both"``).
Expanded targets that share (wavelength, AOI, side) share one evaluation point, because one
evaluation returns R, T, A for both polarisations.

A *beam* is a named operating condition (default: the target name).  Targets on the same beam
share wavelength / AOI / side / polarisation; a target that leaves a condition field ``None``
inherits it from the first target of its beam.  In code, ``Beam`` builds such targets::

    b = Beam("pass", wavelength_nm=(520, 545), points=26)
    targets = [b.min("T", 0.98), b.max("A", 0.001)]

Deviation of one expanded target (value v, scaled by the tolerance):
    eq : (v - value) / tol
    min: max(value - v, 0) / tol      (value is a lower bound)
    max: max(v - value, 0) / tol      (value is an upper bound)
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field, fields, replace

import numpy as np

QUANTITIES = {"R": 0, "T": 1, "A": 2}
KINDS = {"eq": 0, "min": 1, "max": 2}
SIDES = ("front", "back")
CONDITION_FIELDS = ("side", "wavelength_nm", "points", "aoi_deg", "aoi_points", "polarization")
_DEFAULTS = {"side": "front", "aoi_deg": 0.0, "polarization": "unpolarized"}


def polarization_variants(p) -> list[tuple[str, tuple[float, float]]]:
    """Polarisation spec -> [(label, (w_s, w_p))]; 'both' gives two variants."""
    if isinstance(p, str):
        key = p.lower()
        table = {"s": [("s", (1.0, 0.0))], "p": [("p", (0.0, 1.0))],
                 "unpolarized": [("unpol", (0.5, 0.5))], "unpolarised": [("unpol", (0.5, 0.5))],
                 "both": [("s", (1.0, 0.0)), ("p", (0.0, 1.0))]}
        if key not in table:
            raise ValueError(f"unknown polarization {p!r} (s | p | unpolarized | both | azimuth in degrees)")
        return table[key]
    psi = math.radians(float(p))
    return [(f"{float(p):g}deg", (math.sin(psi) ** 2, math.cos(psi) ** 2))]   # 0 deg = p, 90 deg = s


def beam_weights(p) -> tuple[float, float]:
    """Single (w_s, w_p) used to report a beam; 'both' is reported as the s/p average."""
    v = polarization_variants(p)
    return (0.5, 0.5) if len(v) == 2 else v[0][1]


def _num_or_tuple(v):
    if v is None:
        return None
    if isinstance(v, numbers.Real) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, (list, tuple, np.ndarray)):
        return tuple(float(x) for x in v)
    raise ValueError(f"expected a number or a list of numbers, got {v!r}")


def _grid(value, n, what: str) -> tuple[float, ...]:
    if isinstance(value, numbers.Real):
        return (float(value),)
    vals = list(value)
    if n is not None:
        if len(vals) != 2 or n < 1:
            raise ValueError(f"{what}: a range is [start, stop] with points >= 1")
        if vals[1] < vals[0]:
            raise ValueError(f"{what}: range start must not exceed stop")
        return tuple(np.linspace(vals[0], vals[1], int(n)).tolist()) if n > 1 else (0.5 * (vals[0] + vals[1]),)
    if not vals:
        raise ValueError(f"{what}: empty list")
    return tuple(vals)                                # explicit list of values


@dataclass(frozen=True)
class Target:
    """One requirement on one operating condition (see module docstring).

    ``wavelength_nm`` / ``aoi_deg`` are a number, an explicit list, or ``(start, stop)`` together
    with ``points`` / ``aoi_points``.  Condition fields left ``None`` are inherited from the beam
    (or take defaults: side 'front', aoi 0 deg, polarization 'unpolarized')."""
    name: str
    quantity: str
    value: float
    wavelength_nm: float | tuple[float, ...] | None = None
    points: int | None = None
    aoi_deg: float | tuple[float, ...] | None = None
    aoi_points: int | None = None
    polarization: str | float | None = None
    side: str | None = None
    kind: str = "eq"
    tolerance: float = 1.0
    weight: float = 1.0
    beam: str | None = None

    def __post_init__(self):
        s = object.__setattr__
        s(self, "name", str(self.name))
        q = str(self.quantity).upper()
        if q not in QUANTITIES:
            raise ValueError(f"target {self.name!r}: quantity must be one of {tuple(QUANTITIES)}")
        s(self, "quantity", q)
        kind = str(self.kind).lower()
        if kind not in KINDS:
            raise ValueError(f"target {self.name!r}: kind must be one of {tuple(KINDS)}")
        s(self, "kind", kind)
        s(self, "value", float(self.value))
        s(self, "tolerance", float(self.tolerance))
        s(self, "weight", float(self.weight))
        if self.tolerance <= 0 or self.weight < 0:
            raise ValueError(f"target {self.name!r}: tolerance must be > 0 and weight >= 0")
        if self.side is not None:
            s(self, "side", str(self.side).lower())
            if self.side not in SIDES:
                raise ValueError(f"target {self.name!r}: side must be one of {SIDES}")
        if self.polarization is not None:
            pol = self.polarization if isinstance(self.polarization, str) else float(self.polarization)
            polarization_variants(pol)
            s(self, "polarization", pol)
        for f in ("wavelength_nm", "aoi_deg"):
            try:
                s(self, f, _num_or_tuple(getattr(self, f)))
            except ValueError as exc:
                raise ValueError(f"target {self.name!r}: {f}: {exc}") from None
        for f in ("points", "aoi_points"):
            v = getattr(self, f)
            if v is not None:
                s(self, f, int(v))
        if self.beam is not None:
            s(self, "beam", str(self.beam))

    # ---- available once resolved (all condition fields set) ----
    @property
    def resolved(self) -> bool:
        return all(getattr(self, f) is not None for f in ("side", "wavelength_nm", "aoi_deg", "polarization", "beam"))

    def _need_resolved(self):
        if not self.resolved:
            raise ValueError(f"target {self.name!r} is not resolved (use it inside a FilterSystem)")

    @property
    def wavelengths_nm(self) -> tuple[float, ...]:
        self._need_resolved()
        return _grid(self.wavelength_nm, self.points, f"target {self.name!r} wavelength_nm")

    @property
    def aois_deg(self) -> tuple[float, ...]:
        self._need_resolved()
        return _grid(self.aoi_deg, self.aoi_points, f"target {self.name!r} aoi_deg")

    @property
    def n_points(self) -> int:
        return len(self.wavelengths_nm) * len(self.aois_deg) * len(polarization_variants(self.polarization))

    @property
    def center_wavelength(self) -> float:
        w = self.wavelengths_nm
        return 0.5 * (min(w) + max(w))

    @property
    def center_aoi(self) -> float:
        a = self.aois_deg
        return 0.5 * (min(a) + max(a))

    def to_dict(self) -> dict:
        """TOML-row representation (``None`` fields omitted, ranges as lists)."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None:
                out[f.name] = list(v) if isinstance(v, tuple) else v
        return out


TargetRow = Target          # backwards-compatible name


@dataclass(frozen=True)
class Beam:
    """Named operating condition; ``.eq/.min/.max`` create targets on it."""
    label: str
    wavelength_nm: float | tuple[float, ...]
    points: int | None = None
    aoi_deg: float | tuple[float, ...] = 0.0
    aoi_points: int | None = None
    polarization: str | float = "unpolarized"
    side: str = "front"

    def target(self, quantity: str, value: float, kind: str = "eq", name: str | None = None, **kw) -> Target:
        return Target(name=name or f"{str(quantity).upper()}@{self.label}", quantity=quantity, value=value, kind=kind,
                      beam=self.label, side=self.side, wavelength_nm=self.wavelength_nm, points=self.points,
                      aoi_deg=self.aoi_deg, aoi_points=self.aoi_points, polarization=self.polarization, **kw)

    def eq(self, quantity: str, value: float, name: str | None = None, **kw) -> Target:
        return self.target(quantity, value, "eq", name, **kw)

    def min(self, quantity: str, value: float, name: str | None = None, **kw) -> Target:
        """quantity >= value"""
        return self.target(quantity, value, "min", name, **kw)

    def max(self, quantity: str, value: float, name: str | None = None, **kw) -> Target:
        """quantity <= value"""
        return self.target(quantity, value, "max", name, **kw)


def alternating(a: str, b: str, n_layers: int) -> list[str]:
    """[a, b, a, b, ...] with n_layers entries (layer lists are written ambient -> substrate)."""
    return [a if i % 2 == 0 else b for i in range(int(n_layers))]


def resolve_targets(targets) -> tuple[Target, ...]:
    """Fill beam labels, inherit condition fields within beams, apply defaults and validate."""
    if not targets:
        raise ValueError("at least one target is required")
    out, names, beams = [], set(), {}
    for t in targets:
        if not isinstance(t, Target):
            raise TypeError(f"expected Target, got {type(t).__name__}")
        if t.name in names:
            raise ValueError(f"duplicate target name {t.name!r}")
        names.add(t.name)
        beam = t.beam or t.name
        spec = dict(beams.get(beam, {}))
        for f in CONDITION_FIELDS:
            v = getattr(t, f)
            if v is not None:
                if beam in beams and beams[beam].get(f) is not None and beams[beam][f] != v:
                    raise ValueError(f"target {t.name!r}: {f} = {v!r} conflicts with beam {beam!r} ({beams[beam][f]!r})")
                spec[f] = v
        if spec.get("wavelength_nm") is None:
            raise ValueError(f"target {t.name!r}: wavelength_nm missing")
        beams.setdefault(beam, spec)
        beams[beam].update(spec)
        full = {f: spec.get(f, _DEFAULTS.get(f)) for f in CONDITION_FIELDS}
        r = replace(t, beam=beam, **full)
        r.wavelengths_nm                                  # validate the grids
        if any(a < 0 or a >= 90 for a in r.aois_deg):
            raise ValueError(f"target {t.name!r}: aoi_deg must be in [0, 90)")
        out.append(r)
    return tuple(out)


_TARGET_KEYS = {f.name for f in fields(Target)}


def parse_targets(entries: list[dict]) -> tuple[Target, ...]:
    """TOML ``[[targets]]`` rows -> resolved targets."""
    if not entries:
        raise ValueError("config needs at least one [[targets]] row")
    targets = []
    for i, e in enumerate(entries):
        name = str(e.get("name", f"target{i + 1}"))
        unknown = set(e) - _TARGET_KEYS
        if unknown:
            raise ValueError(f"target {name!r}: unknown keys {sorted(unknown)}")
        for key in ("quantity", "value"):
            if key not in e:
                raise ValueError(f"target {name!r}: {key} missing")
        targets.append(Target(**{**e, "name": name}))
    return resolve_targets(targets)


@dataclass(frozen=True)
class BeamGroup:
    """A beam inside a built target table: its condition and evaluation points."""
    label: str
    side: str
    polarization: str | float
    wavelengths_nm: tuple[float, ...]
    aois_deg: tuple[float, ...]
    points: np.ndarray            # evaluation-point indices of the beam's (lambda, aoi) grid, row-major
    rows: tuple[int, ...]         # targets on this beam

    @property
    def single(self) -> bool:
        return self.points.size == 1

    @property
    def weights(self) -> tuple[float, float]:
        return beam_weights(self.polarization)


@dataclass(frozen=True)
class TargetTable:
    rows: tuple[Target, ...]
    point_lam: np.ndarray         # (P,)
    point_aoi: np.ndarray         # (P,)
    point_front: np.ndarray       # (P,) bool: light arrives on the front side
    t_point: np.ndarray           # (K,) evaluation point of each expanded target
    t_polw: np.ndarray            # (K, 2) (w_s, w_p)
    t_q: np.ndarray               # (K,) 0 R, 1 T, 2 A
    t_kind: np.ndarray            # (K,) 0 eq, 1 min, 2 max
    t_value: np.ndarray
    t_tol: np.ndarray
    t_weight: np.ndarray          # target weight split evenly over its expanded points
    t_row: np.ndarray             # (K,) target index
    t_lam: np.ndarray
    t_aoi: np.ndarray
    t_pol: tuple[str, ...]
    beams: dict = field(default_factory=dict)

    @classmethod
    def build(cls, rows: tuple[Target, ...]) -> "TargetTable":
        key_to_point: dict[tuple, int] = {}
        P_lam, P_aoi, P_front = [], [], []

        def point(lam, aoi, side):
            key = (round(lam, 9), round(aoi, 9), side)
            if key not in key_to_point:
                key_to_point[key] = len(P_lam)
                P_lam.append(lam)
                P_aoi.append(aoi)
                P_front.append(side == "front")
            return key_to_point[key]

        cols = {k: [] for k in ("point", "polw", "q", "kind", "value", "tol", "weight", "row", "lam", "aoi", "pol")}
        beam_rows: dict[str, list[int]] = {}
        for r_idx, r in enumerate(rows):
            beam_rows.setdefault(r.beam, []).append(r_idx)
            w = r.weight / r.n_points
            for pol_label, polw in polarization_variants(r.polarization):
                for lam in r.wavelengths_nm:
                    for aoi in r.aois_deg:
                        for k, v in (("point", point(lam, aoi, r.side)), ("polw", polw), ("q", QUANTITIES[r.quantity]),
                                     ("kind", KINDS[r.kind]), ("value", r.value), ("tol", r.tolerance), ("weight", w),
                                     ("row", r_idx), ("lam", lam), ("aoi", aoi), ("pol", pol_label)):
                            cols[k].append(v)
        if sum(cols["weight"]) <= 0:
            raise ValueError("all target weights are zero")
        beams = {}
        for label, r_idx in beam_rows.items():
            r0 = rows[r_idx[0]]
            pts = np.array([point(l, a, r0.side) for l in r0.wavelengths_nm for a in r0.aois_deg])
            beams[label] = BeamGroup(label, r0.side, r0.polarization, r0.wavelengths_nm, r0.aois_deg, pts, tuple(r_idx))
        a = np.asarray
        return cls(rows=rows, point_lam=a(P_lam, float), point_aoi=a(P_aoi, float), point_front=a(P_front, bool),
                   t_point=a(cols["point"], int), t_polw=a(cols["polw"], float), t_q=a(cols["q"], int),
                   t_kind=a(cols["kind"], int), t_value=a(cols["value"], float), t_tol=a(cols["tol"], float),
                   t_weight=a(cols["weight"], float), t_row=a(cols["row"], int), t_lam=a(cols["lam"], float),
                   t_aoi=a(cols["aoi"], float), t_pol=tuple(cols["pol"]), beams=beams)

    @property
    def n_points(self) -> int:
        return self.point_lam.size

    def parse_term(self, term: str) -> tuple[str, str]:
        """'T@1' -> ('T', beam '1'); the beam must be a single point."""
        q, _, beam = term.partition("@")
        q = q.upper()
        if q not in QUANTITIES or beam not in self.beams:
            raise ValueError(f"report term {term!r}: expected <R|T|A>@<beam>, beams: {sorted(self.beams)}")
        if not self.beams[beam].single:
            raise ValueError(f"report term {term!r}: beam {beam!r} is a range, sums need single-point beams")
        return q, beam


def deviation(values, kind, value, tol, xp=np):
    """Scaled deviation per expanded target (works with numpy or jax.numpy)."""
    d_eq = (values - value) / tol
    d_min = xp.maximum(value - values, 0.0) / tol
    d_max = xp.maximum(values - value, 0.0) / tol
    return xp.where(kind == 0, d_eq, xp.where(kind == 1, d_min, d_max))
