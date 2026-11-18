"""Target table: the design specification, as in thin-film design software.

Each ``[[targets]]`` row is an operating condition plus a requirement::

    (wavelength or range, AOI or range, polarisation, side) -> quantity R|T|A, kind eq|min|max, value, tolerance, weight

Rows are expanded over their wavelength x AOI grid (and s/p for ``polarization = "both"``).
Expanded targets that share (wavelength, AOI, side) share one evaluation point, because one
evaluation returns R, T, A for both polarisations.

A *beam* is a named operating condition (default: the row name).  Rows with the same beam
share wavelength / AOI / side / polarisation (later rows inherit omitted fields), so several
requirements can be put on one beam and reports can show R, T, A per beam.

Deviation of one expanded target (value v, scaled by the tolerance):
    eq : (v - value) / tol
    min: max(value - v, 0) / tol      (value is a lower bound)
    max: max(v - value, 0) / tol      (value is an upper bound)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

QUANTITIES = {"R": 0, "T": 1, "A": 2}
KINDS = {"eq": 0, "min": 1, "max": 2}
SIDES = ("front", "back")
_BEAM_FIELDS = ("side", "wavelength_nm", "points", "aoi_deg", "aoi_points", "polarization")


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


def _grid(value, n, what: str) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        return (float(value),)
    vals = [float(v) for v in value]
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
class TargetRow:
    name: str
    beam: str
    side: str
    wavelengths_nm: tuple[float, ...]
    aois_deg: tuple[float, ...]
    polarization: str | float
    quantity: str
    kind: str
    value: float
    tolerance: float
    weight: float

    @property
    def n_points(self) -> int:
        return len(self.wavelengths_nm) * len(self.aois_deg) * len(polarization_variants(self.polarization))

    @property
    def center_wavelength(self) -> float:
        return 0.5 * (min(self.wavelengths_nm) + max(self.wavelengths_nm))

    @property
    def center_aoi(self) -> float:
        return 0.5 * (min(self.aois_deg) + max(self.aois_deg))


def parse_targets(entries: list[dict]) -> tuple[TargetRow, ...]:
    if not entries:
        raise ValueError("config needs at least one [[targets]] row")
    rows, names, beams = [], set(), {}
    for i, e in enumerate(entries):
        name = str(e.get("name", f"target{i + 1}"))
        if name in names:
            raise ValueError(f"duplicate target name {name!r}")
        names.add(name)
        beam = str(e.get("beam", name))
        spec = dict(beams.get(beam, {}))
        for f in _BEAM_FIELDS:
            if f in e:
                if beam in beams and f in beams[beam] and beams[beam][f] != e[f]:
                    raise ValueError(f"target {name!r}: {f} = {e[f]!r} conflicts with beam {beam!r} ({beams[beam][f]!r})")
                spec[f] = e[f]
        if "wavelength_nm" not in spec:
            raise ValueError(f"target {name!r}: wavelength_nm missing")
        beams[beam] = spec
        side = spec.get("side", "front")
        if side not in SIDES:
            raise ValueError(f"target {name!r}: side must be one of {SIDES}")
        q = str(e.get("quantity", "")).upper()
        if q not in QUANTITIES:
            raise ValueError(f"target {name!r}: quantity must be one of {tuple(QUANTITIES)}")
        kind = str(e.get("kind", "eq")).lower()
        if kind not in KINDS:
            raise ValueError(f"target {name!r}: kind must be one of {tuple(KINDS)}")
        if "value" not in e:
            raise ValueError(f"target {name!r}: value missing")
        tol, w = float(e.get("tolerance", 1.0)), float(e.get("weight", 1.0))
        if tol <= 0 or w < 0:
            raise ValueError(f"target {name!r}: tolerance must be > 0 and weight >= 0")
        pol = spec.get("polarization", "unpolarized")
        polarization_variants(pol)                       # validate
        rows.append(TargetRow(
            name=name, beam=beam, side=side,
            wavelengths_nm=_grid(spec["wavelength_nm"], spec.get("points"), f"target {name!r} wavelength_nm"),
            aois_deg=_grid(spec.get("aoi_deg", 0.0), spec.get("aoi_points"), f"target {name!r} aoi_deg"),
            polarization=pol if isinstance(pol, str) else float(pol), quantity=q, kind=kind,
            value=float(e["value"]), tolerance=tol, weight=w))
        if any(a < 0 or a >= 90 for a in rows[-1].aois_deg):
            raise ValueError(f"target {name!r}: aoi_deg must be in [0, 90)")
    return tuple(rows)


@dataclass(frozen=True)
class Beam:
    label: str
    side: str
    polarization: str | float
    wavelengths_nm: tuple[float, ...]
    aois_deg: tuple[float, ...]
    points: np.ndarray            # evaluation-point indices of the beam's (lambda, aoi) grid, row-major
    rows: tuple[int, ...]         # target rows on this beam

    @property
    def single(self) -> bool:
        return self.points.size == 1

    @property
    def weights(self) -> tuple[float, float]:
        return beam_weights(self.polarization)


@dataclass(frozen=True)
class TargetTable:
    rows: tuple[TargetRow, ...]
    point_lam: np.ndarray         # (P,)
    point_aoi: np.ndarray         # (P,)
    point_front: np.ndarray       # (P,) bool: light arrives on the front side
    t_point: np.ndarray           # (K,) evaluation point of each expanded target
    t_polw: np.ndarray            # (K, 2) (w_s, w_p)
    t_q: np.ndarray               # (K,) 0 R, 1 T, 2 A
    t_kind: np.ndarray            # (K,) 0 eq, 1 min, 2 max
    t_value: np.ndarray
    t_tol: np.ndarray
    t_weight: np.ndarray          # row weight split evenly over the row's expanded targets
    t_row: np.ndarray             # (K,) row index
    t_lam: np.ndarray
    t_aoi: np.ndarray
    t_pol: tuple[str, ...]
    beams: dict = field(default_factory=dict)

    @classmethod
    def build(cls, rows: tuple[TargetRow, ...]) -> "TargetTable":
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
            beams[label] = Beam(label, r0.side, r0.polarization, r0.wavelengths_nm, r0.aois_deg, pts, tuple(r_idx))
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
