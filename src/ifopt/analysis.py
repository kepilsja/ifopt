"""Analysis computations on design vectors: per-beam spectra vs wavelength and angle, thickness Monte Carlo.
Plots live in ``plotting.py``; the public wrappers in ``api.py``."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .config import Config
from .objectives import make_merit
from .plate import Model, evaluate, make_points, mix
from .targets import deviation


def material_window(cfg: Config) -> tuple[float, float]:
    """Wavelength interval (nm) in which every material of the system is tabulated."""
    names = {cfg.ambient, cfg.exit_material, *cfg.front.materials, *cfg.back.materials}
    mats = [cfg.library.get(n) for n in names]
    return max(m.wl_min_nm for m in mats), min(m.wl_max_nm for m in mats)


def _curve(cfg: Config, x, lam, aoi, front: bool, polarization):
    from .targets import beam_weights
    pts = make_points(cfg, lam, aoi, front)
    res = evaluate(*cfg.split(x), pts, cfg.mode)
    return np.asarray(res), np.asarray(mix(res, jnp.asarray(beam_weights(polarization))))


def beam_spectrum(cfg: Config, x, label: str, n: int = 401):
    b = cfg.table.beams[label]
    lo, hi = min(b.wavelengths_nm), max(b.wavelengths_nm)
    margin = 50.0 if hi == lo else max(0.15 * (hi - lo), 5.0)
    wlo, whi = material_window(cfg)
    lam = np.linspace(max(lo - margin, wlo), min(hi + margin, whi), n)
    aoi = 0.5 * (min(b.aois_deg) + max(b.aois_deg))
    per, mixed = _curve(cfg, x, lam, aoi, b.side == "front", b.polarization)
    return lam, per, mixed


def beam_vs_angle(cfg: Config, x, label: str, n: int = 201):
    b = cfg.table.beams[label]
    a0 = 0.5 * (min(b.aois_deg) + max(b.aois_deg))
    lo = max(0.0, min(min(b.aois_deg), a0 - 5.0))
    hi = min(89.0, max(max(b.aois_deg), a0 + 5.0, lo + 10.0))
    aoi = np.linspace(lo, hi, n)
    lam = 0.5 * (min(b.wavelengths_nm) + max(b.wavelengths_nm))
    per, mixed = _curve(cfg, x, lam, aoi, b.side == "front", b.polarization)
    return aoi, per, mixed


def thickness_monte_carlo(cfg: Config, x, rel_sigma: float = 0.01, n: int = 2000, seed: int = 1):
    """Gaussian relative thickness errors on the design vector.
    Returns (merit samples (n,), per-row worst |deviation| samples (n, rows))."""
    model = Model(cfg)
    merit, values = make_merit(model)
    t = cfg.table
    rng = np.random.default_rng(seed)
    X = np.clip(np.asarray(x)[None, :] * (1 + rel_sigma * rng.standard_normal((n, len(x)))), 0.0, None)
    m = np.asarray(jax.jit(jax.vmap(merit))(jnp.asarray(X)))
    v = np.asarray(jax.jit(jax.vmap(values))(jnp.asarray(X)))                      # (n, K)
    d = np.abs(deviation(v, t.t_kind[None], t.t_value[None], t.t_tol[None]))
    worst = np.stack([d[:, t.t_row == i].max(axis=1) for i in range(len(t.rows))], axis=1)
    return m, worst
