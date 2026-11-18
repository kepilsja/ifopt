"""Merit functions (minimised) over selected plate coefficients, e.g. maximise T1 and R2."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .config import Config
from .geometry import parse_coefficient
from .plate import PlateModel, mix

_Q = {"R": 0, "T": 1, "A": 2}


def make_merit(cfg: Config, model: PlateModel):
    ob = cfg.objective
    idx = [(s - 1, _Q[q]) for s, q in map(parse_coefficient, ob.maximize)]
    src = jnp.array([i for i, _ in idx])
    qty = jnp.array([j for _, j in idx])
    w = jnp.asarray(ob.weights, float)
    pts = model.points

    def selected(x):
        from .plate import evaluate_points
        res = evaluate_points(*cfg.split(x), pts)         # (2, 2, 3)
        if ob.both_polarizations:
            # each target must hold for s and p separately -> 2x as many terms
            P = jnp.concatenate([res[src, 0, qty], res[src, 1, qty]])
            ww = jnp.concatenate([w, w])
        else:
            P = mix(res, pts.w)[src, qty]
            ww = w
        return P, ww

    def merit(x):
        P, ww = selected(x)
        if ob.form == "least_squares":
            return jnp.sum(ww * (1.0 - P) ** 2)
        if ob.form == "log":
            return -jnp.sum(ww * jnp.log(jnp.clip(P, 1e-12, None)))
        if ob.form == "softmin":
            tau = ob.softmin_temperature
            return tau * jnp.log(jnp.sum(ww * jnp.exp(-P / tau)) / jnp.sum(ww))   # = -softmin(P)
        raise ValueError(f"unknown objective form {ob.form!r}")

    return merit, selected


def _quarter_wave(cfg: Config, materials, lam: float) -> np.ndarray:
    lib = cfg.library
    kx = lib.nk(cfg.ambient, lam)[0].real * np.sin(np.radians(cfg.aoi_deg))
    n = np.array([lib.nk(m, lam)[0] for m in materials], dtype=complex)
    return lam / (4 * np.sqrt(n * n - kx * kx).real)


def quarter_wave_seed(cfg: Config) -> np.ndarray:
    """Seed design vector.

    Front: quarter-wave stack at 45 deg for the wavelength that has to be *reflected* (if any).
    For a T1/R2 combiner (532/1064) the QW stack at 1064 nm has half-wave layers at 532 nm,
    i.e. it is nearly transparent there - a good starting point.
    Back (if optimised): quarter-wave layers at the wavelength that has to be *transmitted*
    (an anti-reflection starting point)."""
    targets = [parse_coefficient(c) for c in cfg.objective.maximize]
    refl = [s for s, q in targets if q == "R"]
    trans = [s for s, q in targets if q == "T"]
    lam_r = cfg.sources[refl[0] if refl else targets[0][0]].wavelength_nm
    lam_t = cfg.sources[trans[0] if trans else targets[0][0]].wavelength_nm
    lo, hi = cfg.design_bounds()
    parts = [_quarter_wave(cfg, cfg.coating.materials, lam_r)]
    if cfg.back_optimized:
        parts.append(_quarter_wave(cfg, cfg.back.materials, lam_t))
    return np.clip(np.concatenate(parts), lo, hi)
