"""Public operations: evaluate, optimise, sweep, Monte Carlo, report, quick stack solves.

All functions accept a ``Design`` or a plain design vector, run in 64-bit JAX mode (even if the
caller switched x64 off) and return frozen dataclasses holding floats and NumPy arrays.
"""
from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import analysis
from .config import FilterSystem, OptimizerSettings
from .design import Design, as_x
from .objectives import figure_of_merit, make_merit, row_summary
from .plate import Model, evaluate as _evaluate_points, evaluate_with_engine, make_points, mix
from .targets import beam_weights, deviation

logger = logging.getLogger(__name__)
F32_LIMIT, F64_LIMIT = 88.72, 709.78
_SYM = {"eq": "=", "min": ">=", "max": "<="}


def x64(fn):
    """Run ``fn`` with JAX 64-bit mode enabled (results stay float64 whatever the global setting)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with jax.enable_x64(True):
            return fn(*args, **kwargs)
    return wrapper


# ------------------------------------------------------------------------------ results --
@dataclass(frozen=True)
class BeamResult:
    label: str
    side: str
    polarization: str | float
    wavelengths_nm: np.ndarray        # (n,) grid points of the beam (row-major over wavelength x AOI)
    aois_deg: np.ndarray              # (n,)
    per_pol: np.ndarray               # (n, 2 [s, p], 3 [R, T, A])
    mixed: np.ndarray                 # (n, 3) with the beam's polarisation

    R = property(lambda self: self.mixed[:, 0])
    T = property(lambda self: self.mixed[:, 1])
    A = property(lambda self: self.mixed[:, 2])

    @property
    def single(self) -> bool:
        return self.mixed.shape[0] == 1


@dataclass(frozen=True)
class TargetResult:
    name: str
    beam: str
    quantity: str
    kind: str
    value: float
    achieved_min: float
    achieved_max: float
    worst_deviation: float
    share: float                      # fraction of the merit contributed by this target
    points: int

    @property
    def met(self) -> bool | None:
        """For min/max targets: requirement met at every point; None for eq targets."""
        return None if self.kind == "eq" else self.worst_deviation <= 1e-12


@dataclass(frozen=True)
class Evaluation:
    system: FilterSystem = field(repr=False)
    design: Design = field(repr=False)
    response: np.ndarray = field(repr=False)          # (P, 2, 3) at the target table's points
    beams: dict = field(repr=False)
    targets: tuple[TargetResult, ...]
    merit: float
    mf: float | None
    sums: dict
    envelope: dict | None = field(default=None, repr=False)     # beam -> (min (3,), max (3,)), plate modes
    ceiling: dict | None = field(default=None, repr=False)      # beam -> T bound, incoherent mode
    cross_check: dict = field(default_factory=dict, repr=False)  # engine -> (P, 2, 3) |diff| (NaN = not sampled)
    sum_im_delta: np.ndarray = field(default=None, repr=False)
    label: str = ""

    def report(self) -> str:
        return format_report(self)

    def to_dict(self) -> dict:
        return {"label": self.label, "merit": self.merit, "mf": self.mf, "sums": dict(self.sums),
                "design": self.design.to_dict(self.system),
                "targets": [t.__dict__ | {"met": t.met} for t in self.targets],
                "beams": {k: {"side": b.side, "polarization": b.polarization,
                              "wavelengths_nm": b.wavelengths_nm.tolist(), "aois_deg": b.aois_deg.tolist(),
                              "R": b.R.tolist(), "T": b.T.tolist(), "A": b.A.tolist()} for k, b in self.beams.items()}}

    def to_pandas(self):
        """One row per expanded target point."""
        import pandas as pd
        t, sysm = self.system.table, self.system
        values = np.einsum("kp,kp->k", t.t_polw, self.response[t.t_point, :, t.t_q])
        d = deviation(values, t.t_kind, t.t_value, t.t_tol)
        return pd.DataFrame({
            "target": [sysm.targets[i].name for i in t.t_row], "beam": [sysm.targets[i].beam for i in t.t_row],
            "wavelength_nm": t.t_lam, "aoi_deg": t.t_aoi, "polarization": list(t.t_pol),
            "quantity": [sysm.targets[i].quantity for i in t.t_row], "kind": [sysm.targets[i].kind for i in t.t_row],
            "value": t.t_value, "achieved": values, "deviation": d, "weight": t.t_weight})


@dataclass(frozen=True)
class OptimizationResult:
    design: Design
    evaluation: Evaluation
    merit: float
    mf: float | None
    history: dict = field(repr=False)
    candidates: tuple = field(repr=False)             # ((merit, Design), ...) best first
    timings: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Spectrum:
    wavelengths_nm: np.ndarray
    aoi_deg: np.ndarray
    side: str
    polarization: str | float
    per_pol: np.ndarray               # (n, 2, 3)
    mixed: np.ndarray                 # (n, 3)

    R = property(lambda self: self.mixed[:, 0])
    T = property(lambda self: self.mixed[:, 1])
    A = property(lambda self: self.mixed[:, 2])


@dataclass(frozen=True)
class MonteCarlo:
    rel_sigma: float
    merit: np.ndarray                 # (n,)
    mf: np.ndarray                    # (n,)
    worst_deviation: np.ndarray       # (n, targets)
    target_names: tuple[str, ...]
    target_kinds: tuple[str, ...]
    nominal_mf: float | None

    def met_fraction(self, name: str) -> float:
        i = self.target_names.index(name)
        return float(np.mean(self.worst_deviation[:, i] <= 1e-12))

    def summary(self) -> str:
        n = self.merit.size
        lines = [f"Thickness Monte Carlo ({100 * self.rel_sigma:g}% rel. sigma per layer, {n} samples):",
                 f"  MF: nominal {self.nominal_mf:.4e}, median {np.median(self.mf):.4e}, "
                 f"95th pct {np.percentile(self.mf, 95):.4e}"]
        for i, (name, kind) in enumerate(zip(self.target_names, self.target_kinds)):
            w = self.worst_deviation[:, i]
            met = f", requirement met in {100 * self.met_fraction(name):.0f}% of samples" if kind != "eq" else ""
            lines.append(f"  {name}: worst |d| median {np.median(w):.2e}, 95th pct {np.percentile(w, 95):.2e}{met}")
        return "\n".join(lines)


@dataclass(frozen=True)
class StackRT:
    R: float
    T: float
    Rb: float                         # reflectance seen from the exit side
    Tb: float

    @property
    def A(self) -> float:
        return 1.0 - self.R - self.T


# --------------------------------------------------------------------------- operations --
def _beam_results(system: FilterSystem, res: np.ndarray) -> dict:
    out = {}
    t = system.table
    for label, b in t.beams.items():
        per = res[b.points]
        lam = np.repeat(np.asarray(b.wavelengths_nm), len(b.aois_deg))
        aoi = np.tile(np.asarray(b.aois_deg), len(b.wavelengths_nm))
        out[label] = BeamResult(label, b.side, b.polarization, lam, aoi, per,
                                np.einsum("p,npk->nk", np.asarray(b.weights), per))
    return out


@x64
def evaluate(system: FilterSystem, design, *, engines=(), extras: bool = True, max_engine_points: int = 40,
             label: str = "") -> Evaluation:
    """Evaluate a design against the system's targets.

    engines: names of cross-check engines ("tmmax", "reference"), compared on up to
    ``max_engine_points`` evenly spaced points.  extras: fringe envelope and back-face ceiling
    (plate modes)."""
    x = as_x(system, design)
    d = design if isinstance(design, Design) else Design.from_x(system, x)
    model = Model(system)
    t = system.table
    res = np.asarray(model.response(x))
    merit, _ = make_merit(model)
    mv = float(merit(jnp.asarray(x)))
    rows = tuple(TargetResult(**r) for r in row_summary(model, x))
    beams = _beam_results(system, res)
    sums = {}
    for s in system.sums:
        total = 0.0
        for term in s.terms:
            q, beam = t.parse_term(term)
            total += float(beams[beam].mixed[0, "RTA".index(q)])
        sums[s.name] = total
    envelope = ceiling = None
    if extras and system.has_back_face:
        scan = model.coherent_scan(x)
        envelope = {}
        for label_, b in t.beams.items():
            if b.single:
                env = np.einsum("p,npk->nk", np.asarray(b.weights), scan[int(b.points[0])])
                envelope[label_] = (env.min(axis=0), env.max(axis=0))
        ceil = model.back_face_ceiling(x)
        if ceil is not None:
            ceiling = {lb: float(np.asarray(b.weights) @ ceil[int(b.points[0])]) for lb, b in t.beams.items() if b.single}
    cross = {}
    if engines:
        from .engines import ReferenceEngine, TmmaxEngine
        sel = np.arange(t.n_points) if t.n_points <= max_engine_points else \
            np.unique(np.linspace(0, t.n_points - 1, max_engine_points).round().astype(int))
        for name in engines:
            try:
                eng = {"tmmax": TmmaxEngine, "reference": ReferenceEngine}[name]()
                full = np.full((t.n_points, 2, 3), np.nan)
                full[sel] = np.abs(evaluate_with_engine(system, x, eng, sel) - res[sel])
                cross[name] = full
            except Exception as exc:                    # report and continue
                logger.warning("%s cross-check unavailable: %s: %s", name, type(exc).__name__, exc)
    return Evaluation(system=system, design=d, response=res, beams=beams, targets=rows, merit=mv,
                      mf=figure_of_merit(mv, system.objective) if system.objective.form == "power" else None,
                      sums=sums, envelope=envelope, ceiling=ceiling, cross_check=cross,
                      sum_im_delta=model.sum_im_delta(x), label=label)


@x64
def optimize(system: FilterSystem, *, start=None, settings: OptimizerSettings | None = None,
             engines=()) -> OptimizationResult:
    """Optimise the layer thicknesses of ``system`` against its targets.

    start: a Design / vector or a list of them, added to the multi-start pool.
    settings: overrides ``system.optimizer``.  Progress goes to the ``ifopt.optimizer`` logger."""
    from .optimizer import run_optimization
    if settings is not None:
        system = system.replace(optimizer=settings)
    starts = [] if start is None else (list(start) if isinstance(start, (list, tuple)) else [start])
    res = run_optimization(system, [as_x(system, s) for s in starts])
    design = Design.from_x(system, res.d)
    ev = evaluate(system, design, engines=engines, label=f"optimised, merit {res.merit:.3e}")
    return OptimizationResult(design=design, evaluation=ev, merit=res.merit, mf=ev.mf, history=res.history,
                              candidates=tuple((float(v), Design.from_x(system, np.asarray(xx))) for v, xx in res.candidates),
                              timings=res.timings)


@x64
def spectrum(system: FilterSystem, design, *, wavelengths_nm, aoi_deg=0.0, side: str = "front",
             polarization="unpolarized") -> Spectrum:
    """R/T/A on any wavelength and/or AOI grid (broadcast together), independent of the targets."""
    if side not in ("front", "back"):
        raise ValueError("side must be 'front' or 'back'")
    lam, aoi = (np.ravel(a) for a in np.broadcast_arrays(np.asarray(wavelengths_nm, float), np.asarray(aoi_deg, float)))
    pts = make_points(system, lam, aoi, side == "front")
    per = np.asarray(_evaluate_points(*system.split(as_x(system, design)), pts, system.mode))
    return Spectrum(lam, aoi, side, polarization, per, np.einsum("p,npk->nk", np.asarray(beam_weights(polarization)), per))


@x64
def beam_spectrum(system: FilterSystem, design, beam: str, n: int = 401) -> Spectrum:
    """Wavelength sweep around a beam's band (or +-50 nm around a single wavelength)."""
    b = system.table.beams[beam]
    lam, per, mixed = analysis.beam_spectrum(system, as_x(system, design), beam, n)
    aoi = np.full(lam.shape, 0.5 * (min(b.aois_deg) + max(b.aois_deg)))
    return Spectrum(lam, aoi, b.side, b.polarization, per, mixed)


@x64
def angle_scan(system: FilterSystem, design, beam: str, n: int = 201) -> Spectrum:
    """AOI sweep (+-5 deg, or 0-10 deg at normal incidence) at a beam's centre wavelength."""
    b = system.table.beams[beam]
    aoi, per, mixed = analysis.beam_vs_angle(system, as_x(system, design), beam, n)
    lam = np.full(aoi.shape, 0.5 * (min(b.wavelengths_nm) + max(b.wavelengths_nm)))
    return Spectrum(lam, aoi, b.side, b.polarization, per, mixed)


@x64
def monte_carlo(system: FilterSystem, design, rel_sigma: float = 0.01, n: int = 2000, seed: int = 1) -> MonteCarlo:
    """Gaussian relative thickness errors on every optimised layer."""
    x = as_x(system, design)
    m, worst = analysis.thickness_monte_carlo(system, x, rel_sigma=rel_sigma, n=n, seed=seed)
    merit, _ = make_merit(Model(system))
    ob = system.objective
    return MonteCarlo(rel_sigma, m, np.array([figure_of_merit(v, ob) for v in m]), worst,
                      tuple(t.name for t in system.targets), tuple(t.kind for t in system.targets),
                      figure_of_merit(float(merit(jnp.asarray(x))), ob))


def report(system: FilterSystem, design, engines=("tmmax", "reference"), label: str = "") -> str:
    """Text report (the CLI ``evaluate`` output)."""
    return evaluate(system, design, engines=engines, label=label).report()


@x64
def stack_rt(n_layers, d_nm, *, wavelength_nm: float, n_out, n_in=1.0, aoi_deg: float = 0.0,
             polarization="s") -> StackRT:
    """R/T of ambient | layers | exit for given complex indices (no system needed).
    AOI is taken in the incidence medium ``n_in`` (must be lossless)."""
    from .engines.stable_engine import stack_powers
    n_in = complex(n_in)
    if n_in.imag != 0:
        raise ValueError("the incidence medium must be lossless")
    kx = n_in.real * np.sin(np.radians(aoi_deg))
    w = np.asarray(beam_weights(polarization))
    vals = np.zeros(4)
    for pol in (0, 1):
        if w[pol]:
            F = stack_powers(n_in, jnp.asarray(n_layers, complex), complex(n_out), jnp.asarray(d_nm, float),
                             kx, float(wavelength_nm), pol)
            vals += w[pol] * np.array([float(v) for v in F])
    return StackRT(*vals)


# ------------------------------------------------------------------------------- report --
def _z(v) -> float:
    return 0.0 if abs(v) < 5e-13 else float(v)


def _rel(p) -> str:
    try:
        return os.path.relpath(p)
    except ValueError:
        return str(p)


def _span(vals, unit):
    lo, hi = float(np.min(vals)), float(np.max(vals))
    n = len(set(np.round(vals, 9)))
    return f"{lo:g} {unit}" if lo == hi else f"{lo:g}-{hi:g} {unit} ({n} pts)"


def format_report(ev: Evaluation) -> str:
    system, t = ev.system, ev.system.table
    L = []
    out = L.append
    out(f"Config: {_rel(system.source) if system.source else '(built in code)'}")
    if ev.label:
        out(f"Design: {ev.label}")
    sub = system.substrate
    desc = {"incoherent": f"{sub.material} {sub.thickness_mm:g} mm, incoherent",
            "coherent": f"{sub.material} {sub.thickness_mm:g} mm, coherent (etalon)",
            "semi_infinite": f"{sub.material}, semi-infinite (back surface ignored)",
            "none": f"none - exit medium {system.exit_material}"}[system.mode]
    out(f"Ambient {system.ambient}; substrate: {desc}")
    d = ev.design
    out("Front layers (ambient -> substrate): " + ", ".join(f"{m} {v:.2f}" for m, v in zip(system.front.materials, d.front_nm)) + " nm")
    if system.back.materials:
        out(f"Back layers  (ambient -> substrate, {'optimised' if system.back_optimized else 'fixed'}): "
            + ", ".join(f"{m} {v:.2f}" for m, v in zip(system.back.materials, d.back_nm)) + " nm")
    elif system.has_back_face:
        out("Back face: bare")
    out("")
    for label, b in t.beams.items():
        br = ev.beams[label]
        out(f"Beam {label}: {_span(br.wavelengths_nm, 'nm')}, AOI {_span(br.aois_deg, 'deg')}, {b.side} side, "
            f"polarization {b.polarization!r}")
        eng_cols = "".join(f" {'|d ' + n + '|':>13s}" for n in ev.cross_check)
        if b.single:
            p = int(b.points[0])
            env = (ev.envelope or {}).get(label)
            out(f"  {'':3s} {'s':>10s} {'p':>10s} {'beam':>10s}{eng_cols}" + ("   coherent min      max" if env else ""))
            for k, q in enumerate("RTA"):
                row = f"  {q:3s} {_z(br.per_pol[0, 0, k]):10.6f} {_z(br.per_pol[0, 1, k]):10.6f} {_z(br.mixed[0, k]):10.6f}"
                row += "".join(f" {np.nanmax(dd[p, :, k]):13.2e}" if np.isfinite(dd[p]).any() else f" {'-':>13s}"
                               for dd in ev.cross_check.values())
                if env:
                    row += f"   {_z(env[0][k]):12.6f} {_z(env[1][k]):8.6f}"
                out(row)
            if ev.ceiling and label in ev.ceiling:
                out(f"  back-face ceiling (any passive front coating): T <= {ev.ceiling[label]:.6f}")
        else:
            out(f"  {'':3s} {'min':>10s} {'mean':>10s} {'max':>10s}{eng_cols}   (beam polarization)")
            for k, q in enumerate("RTA"):
                row = f"  {q:3s} {_z(br.mixed[:, k].min()):10.6f} {_z(br.mixed[:, k].mean()):10.6f} {_z(br.mixed[:, k].max()):10.6f}"
                for dd in ev.cross_check.values():
                    v = dd[b.points][:, :, k]
                    row += f" {np.nanmax(v):13.2e}" if np.isfinite(v).any() else f" {'-':>13s}"
                out(row)
        out("")
    ob = system.objective
    out(f"{'target':12s} {'beam':8s} {'requirement':16s} {'achieved (min..max)':>24s} {'worst |d|':>10s} {'share':>7s}")
    for r in ev.targets:
        ach = f"{r.achieved_min:.6f}" if r.points == 1 else f"{r.achieved_min:.6f}..{r.achieved_max:.6f}"
        out(f"{r.name:12s} {r.beam:8s} {r.quantity + ' ' + _SYM[r.kind] + ' ' + format(r.value, 'g'):16s} "
            f"{ach:>24s} {r.worst_deviation:10.2e} {100 * r.share:6.1f}%")
    form = f"power, p = {ob.power:g}" if ob.form == "power" else ob.form
    out(f"Merit ({form}): {ev.merit:.4e}" + (f"   MF = {ev.mf:.4e}" if ev.mf is not None else ""))
    for s in system.sums:
        out(f"{s.name} = {' + '.join(s.terms)} = {ev.sums[s.name]:.6f}")
    out(f"Stability: max sum Im(delta) over points = {np.max(ev.sum_im_delta):.3g} (T-matrix overflow at ~{F32_LIMIT} "
        f"in complex64 / ~{F64_LIMIT} in complex128; the S-matrix engine has no limit)")
    if ev.envelope is not None:
        out("Coherent min/max: substrate as an etalon (single-frequency laser, parallel plate); "
            "its phase average equals the incoherent value.")
    return "\n".join(L)
