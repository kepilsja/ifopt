"""Thickness optimisation (fixed material sequence).

Pipeline
  A. multi-start Adam (optax) on unconstrained u, d = lo + (hi - lo) * sigmoid(u); all starts
     vmapped in a single jitted call (quarter-wave seed + jittered copies + uniform random starts)
  B. L-BFGS-B polish (scipy, native box bounds) of the best ``top_k`` designs with exact float64
     gradients from jax.value_and_grad.  L-BFGS-B = limited-memory quasi-Newton with bounds: it
     builds a curvature estimate from recent gradient differences and converges much faster than
     Adam near an optimum, but it is local - hence stage A.
  (optional) global differential evolution evaluating the vmapped merit on whole populations.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import differential_evolution, minimize

from .config import Config
from .objectives import make_merit, quarter_wave_seed
from .plate import PlateModel


@dataclass
class OptResult:
    d: np.ndarray
    merit: float
    selected: dict
    history: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)


def _logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def initial_designs(cfg: Config, n_starts: int, rng: np.random.Generator) -> np.ndarray:
    lo, hi = map(np.asarray, cfg.design_bounds())
    L = lo.size
    seeds = [quarter_wave_seed(cfg)]
    if (x_cfg := cfg.configured_design()) is not None:
        seeds.append(np.clip(x_cfg, lo, hi))
    n_jitter = max(0, n_starts // 4)
    jitter = [np.clip(seeds[0] * rng.normal(1.0, 0.15, L), lo, hi) for _ in range(n_jitter)]
    n_rand = max(0, n_starts - len(seeds) - n_jitter)
    rand = list(rng.uniform(lo, hi, size=(n_rand, L)))
    return np.asarray(seeds + jitter + rand)


def optimize(cfg: Config, log=print) -> OptResult:
    oc = cfg.optimizer
    model = PlateModel(cfg)
    merit, selected = make_merit(cfg, model)
    lo, hi = map(lambda a: jnp.asarray(a, float), cfg.design_bounds())
    span = hi - lo
    rng = np.random.default_rng(oc.seed)
    timings, history = {}, {}

    def d_of_u(u):
        return lo + span * jax.nn.sigmoid(u)

    def merit_u(u):
        return merit(d_of_u(u))

    # ---- stage A: vmapped multi-start Adam -------------------------------------------------
    D0 = initial_designs(cfg, oc.n_starts, rng)
    U0 = jnp.asarray(_logit((D0 - np.asarray(lo)) / np.asarray(span)))
    opt = optax.adam(oc.adam_lr)

    @jax.jit
    def run_adam(U0):
        def single(u0):
            def step(carry, _):
                u, s = carry
                val, g = jax.value_and_grad(merit_u)(u)
                upd, s = opt.update(g, s, u)
                return (optax.apply_updates(u, upd), s), val
            (u, _), hist = jax.lax.scan(step, (u0, opt.init(u0)), None, length=oc.adam_steps)
            return u, hist
        return jax.vmap(single)(U0)

    t0 = time.perf_counter()
    U, hist = run_adam(U0)
    finals = np.asarray(jax.vmap(merit_u)(U))
    timings["adam_s"] = time.perf_counter() - t0
    history["adam_best_curve"] = np.asarray(hist[np.nanargmin(finals)])
    history["adam_final_merits"] = finals
    log(f"[A] Adam: {len(D0)} starts x {oc.adam_steps} steps in {timings['adam_s']:.1f} s; "
        f"best merit {np.nanmin(finals):.3e}, median {np.nanmedian(finals):.3e}")

    candidates = [np.asarray(d_of_u(U[i])) for i in np.argsort(finals)[: oc.top_k]]

    # ---- optional global DE ----------------------------------------------------------------
    if oc.global_de:
        batch = jax.jit(jax.vmap(merit))
        t0 = time.perf_counter()
        de = differential_evolution(lambda X: np.asarray(batch(jnp.asarray(X.T))), list(zip(np.asarray(lo), np.asarray(hi))),
                                    vectorized=True, polish=False, maxiter=oc.de_maxiter, popsize=oc.de_popsize,
                                    seed=oc.seed, tol=1e-10)
        timings["de_s"] = time.perf_counter() - t0
        log(f"[DE] differential evolution: merit {de.fun:.3e} after {de.nit} generations ({timings['de_s']:.1f} s)")
        candidates.insert(0, de.x)

    # ---- stage B: L-BFGS-B polish ----------------------------------------------------------
    vg = jax.jit(jax.value_and_grad(merit))

    def fun(x):
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=float)

    t0 = time.perf_counter()
    polished = []
    for d0 in candidates:
        r = minimize(fun, d0, jac=True, method="L-BFGS-B", bounds=list(zip(np.asarray(lo), np.asarray(hi))),
                     options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-12})
        polished.append((float(r.fun), r.x, int(r.nit)))
    timings["lbfgsb_s"] = time.perf_counter() - t0
    polished.sort(key=lambda t: t[0])
    best_val, best_d, nit = polished[0]
    log(f"[B] L-BFGS-B polish of {len(candidates)} candidates in {timings['lbfgsb_s']:.1f} s; "
        f"best merit {best_val:.3e} ({nit} iterations)")

    P, _ = selected(jnp.asarray(best_d))
    names = cfg.objective.maximize
    if cfg.objective.both_polarizations:
        sel = {f"{n}_s": float(v) for n, v in zip(names, np.asarray(P)[: len(names)])}
        sel |= {f"{n}_p": float(v) for n, v in zip(names, np.asarray(P)[len(names):])}
    else:
        sel = {n: float(v) for n, v in zip(names, np.asarray(P))}
    return OptResult(d=np.asarray(best_d), merit=best_val, selected=sel, history=history,
                     candidates=[(v, x.tolist()) for v, x, _ in polished], timings=timings)
