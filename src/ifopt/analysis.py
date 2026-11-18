"""Post-optimisation analysis: per-beam spectra vs wavelength and angle, thickness Monte Carlo, plots."""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .config import Config
from .objectives import make_merit
from .plate import Model, evaluate, make_points, mix
from .targets import deviation

COLORS = {"R": "C1", "T": "C0", "A": "C2"}
MARK = {"eq": "o", "min": "^", "max": "v"}


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


def plot_design(cfg: Config, x, out_dir: Path, history: dict | None = None) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    beams = list(cfg.table.beams)
    fig, axes = plt.subplots(2, len(beams), figsize=(6 * len(beams), 8), constrained_layout=True, squeeze=False)
    for col, label in enumerate(beams):
        b = cfg.table.beams[label]
        lam, per, m = beam_spectrum(cfg, x, label)
        ax = axes[0, col]
        for k, q in enumerate("RTA"):
            ax.plot(lam, m[:, k], color=COLORS[q], lw=1.6 if q != "A" else 0.9, label=q)
        if cfg.mode in ("incoherent", "coherent") or b.aois_deg != (0.0,):
            ax.plot(lam, per[:, 0, 1], ":", color="C3", lw=0.8, label="T s")
            ax.plot(lam, per[:, 1, 1], "--", color="C4", lw=0.8, label="T p")
        for i in b.rows:
            r = cfg.table.rows[i]
            lo, hi = min(r.wavelengths_nm), max(r.wavelengths_nm)
            if hi > lo:
                ax.hlines(r.value, lo, hi, color=COLORS[r.quantity], lw=3, alpha=0.35)
            ax.plot([0.5 * (lo + hi)], [r.value], MARK[r.kind], color=COLORS[r.quantity], ms=8, mec="k",
                    label=f"{r.name}: {r.quantity} {dict(eq='=', min='>=', max='<=')[r.kind]} {r.value:g}")
        ax.set(xlabel="wavelength (nm)", ylabel="power fraction", ylim=(-0.02, 1.02),
               title=f"beam {label}: {b.side} side, AOI {0.5 * (min(b.aois_deg) + max(b.aois_deg)):g} deg, pol {b.polarization}")
        ax.legend(fontsize=7)
        aoi, per, m = beam_vs_angle(cfg, x, label)
        ax = axes[1, col]
        for k, q in enumerate("RT"):
            ax.plot(aoi, m[:, k], color=COLORS[q], label=q)
        ax.set(xlabel="angle of incidence (deg)", ylabel="power fraction", ylim=(-0.02, 1.02),
               title=f"beam {label} at {0.5 * (min(b.wavelengths_nm) + max(b.wavelengths_nm)):g} nm")
        ax.legend(fontsize=7)
    p = out_dir / "spectra.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    files.append(p)

    d_front, d_back = (np.asarray(a) for a in cfg.split(x))
    faces = [("front", cfg.front.materials, d_front)]
    if cfg.back.materials:
        faces.append(("back", cfg.back.materials, d_back))
    fig, ax = plt.subplots(figsize=(9, 1.5 + 1.2 * len(faces)), constrained_layout=True)
    colors = {m: f"C{i}" for i, m in enumerate(dict.fromkeys(cfg.front.materials + cfg.back.materials))}
    for row, (_, mats, th) in enumerate(faces):
        pos = 0.0
        for m_, t_ in zip(mats, th):
            ax.barh(row, t_, left=pos, color=colors[m_], edgecolor="k", lw=0.3)
            pos += t_
    for m_, c in colors.items():
        ax.barh(0, 0, color=c, label=m_)
    ax.set(xlabel="depth from ambient (nm)", yticks=list(range(len(faces))), yticklabels=[f[0] for f in faces],
           title=f"layer stacks (ambient -> substrate), substrate mode: {cfg.mode}")
    ax.legend(ncol=len(colors), fontsize=8, loc="upper right")
    p = out_dir / "stack.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    files.append(p)

    if history and "adam_best_curve" in history:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.semilogy(history["adam_best_curve"])
        ax.set(xlabel="Adam step", ylabel="merit", title="convergence of the best start")
        p = out_dir / "convergence.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        files.append(p)
    return files
