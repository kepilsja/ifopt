"""Merit function over the target table (thin-film design convention) and starting designs.

Per expanded target k: scaled deviation d_k (``targets.deviation``) and weight w_k (row weight split
over the row's points).  Forms (what is minimised):

  power   : sum w |d|^p / sum w            reported MF = (.)^(1/p)   (p = 2: least squares)
  minimax : tau log( sum w exp(d^2/tau) / sum w )                    smooth worst case
  log     : -sum w log(1 - |d|) / sum w                              (= -log P for targets P -> 1)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .config import Config, ObjectiveCfg
from .plate import Model
from .targets import deviation


def _abs_pow(d, p: float):
    d2 = d * d
    if p == 2.0:
        return d2
    pos = d2 > 0
    return jnp.where(pos, jnp.where(pos, d2, 1.0) ** (p / 2), 0.0)   # double-where: finite gradients at 0


def merit_from_deviation(d, w, ob: ObjectiveCfg):
    wsum = jnp.sum(w)
    if ob.form == "power":
        return jnp.sum(w * _abs_pow(d, ob.power)) / wsum
    if ob.form == "minimax":
        tau = ob.temperature
        return tau * (jax.nn.logsumexp(d * d / tau, b=w) - jnp.log(wsum))
    if ob.form == "log":
        return -jnp.sum(w * jnp.log(jnp.clip(1.0 - jnp.abs(d), 1e-12, None))) / wsum
    raise ValueError(f"unknown objective form {ob.form!r}")


def figure_of_merit(merit_value: float, ob: ObjectiveCfg) -> float:
    return float(merit_value) ** (1.0 / ob.power) if ob.form == "power" else float(merit_value)


def make_merit(model: Model):
    """Returns (merit(x), values(x)); both jit-/grad-compatible."""
    t, ob = model.table, model.cfg.objective
    kind, value, tol, w = (jnp.asarray(a) for a in (t.t_kind, t.t_value, t.t_tol, t.t_weight))

    def values(x):
        return model.target_values(x)

    def merit(x):
        return merit_from_deviation(deviation(values(x), kind, value, tol, xp=jnp), w, ob)

    return merit, values


def row_summary(model: Model, x) -> list[dict]:
    """Per target row: achieved values (min/max over its points), worst scaled deviation, share of merit."""
    t, ob = model.table, model.cfg.objective
    v = np.asarray(model.target_values(x))
    d = np.asarray(deviation(v, t.t_kind, t.t_value, t.t_tol))
    contrib = t.t_weight * np.abs(d) ** (ob.power if ob.form == "power" else 2)
    total = contrib.sum() or 1.0
    out = []
    for i, r in enumerate(t.rows):
        m = t.t_row == i
        out.append({"name": r.name, "beam": r.beam, "quantity": r.quantity, "kind": r.kind, "value": r.value,
                    "achieved_min": float(v[m].min()), "achieved_max": float(v[m].max()),
                    "worst_deviation": float(np.abs(d[m]).max()), "share": float(contrib[m].sum() / total),
                    "points": int(m.sum())})
    return out


def _quarter_wave(cfg: Config, materials, lam: float, aoi_deg: float) -> np.ndarray:
    lib = cfg.library
    kx = lib.nk(cfg.ambient, lam)[0].real * np.sin(np.radians(aoi_deg))
    n = np.array([lib.nk(m, lam)[0] for m in materials], dtype=complex)
    return lam / (4 * np.sqrt(n * n - kx * kx).real)


def quarter_wave_seed(cfg: Config) -> np.ndarray:
    """Seed design vector.

    Front: quarter-wave stack at the centre wavelength / AOI of the highest-weight 'reflect' row
    (R with eq/min and value >= 0.5); e.g. for a 532/1064 combiner the QW stack at 1064 nm has
    half-wave layers at 532 nm, i.e. it is nearly transparent there.
    Back (if optimised): quarter-wave layers at the highest-weight 'transmit' row (an AR start)."""
    rows = cfg.targets

    def pick(q):
        cand = [r for r in rows if r.quantity == q and r.kind in ("eq", "min") and r.value >= 0.5]
        return max(cand, key=lambda r: r.weight) if cand else rows[0]

    r_ref, r_tr = pick("R"), pick("T")
    lo, hi = cfg.design_bounds()
    parts = [_quarter_wave(cfg, cfg.front.materials, r_ref.center_wavelength, r_ref.center_aoi)]
    if cfg.back_optimized:
        parts.append(_quarter_wave(cfg, cfg.back.materials, r_tr.center_wavelength, r_tr.center_aoi))
    return np.clip(np.concatenate(parts), lo, hi)
