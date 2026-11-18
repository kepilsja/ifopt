"""Post-optimisation analysis: spectra vs wavelength and angle, thickness Monte Carlo, plots."""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .config import Config
from .plate import evaluate_points, make_points, mix


def spectrum_vs_wavelength(cfg: Config, d, source: int, half_width_nm: float = 50.0, n: int = 401):
    lam0 = cfg.sources[source].wavelength_nm
    lam = np.linspace(lam0 - half_width_nm, lam0 + half_width_nm, n)
    pts = make_points(cfg, source, lam_nm=lam)
    res = evaluate_points(*cfg.split(d), pts)          # (n, 2, 3)
    return lam, np.asarray(res), np.asarray(mix(res, pts.w))


def spectrum_vs_angle(cfg: Config, d, source: int, aoi_range=(40.0, 50.0), n: int = 201):
    aoi = np.linspace(*aoi_range, n)
    pts = make_points(cfg, source, aoi_deg=aoi)
    res = evaluate_points(*cfg.split(d), pts)
    return aoi, np.asarray(res), np.asarray(mix(res, pts.w))


def thickness_monte_carlo(cfg: Config, d, rel_sigma: float = 0.01, n: int = 2000, seed: int = 1):
    """Gaussian relative thickness errors (sigma = rel_sigma); returns (n, 2 sources, 3) mixed."""
    from .plate import PlateModel
    model = PlateModel(cfg)
    rng = np.random.default_rng(seed)
    D = np.asarray(d)[None, :] * (1 + rel_sigma * rng.standard_normal((n, len(d))))
    D = np.clip(D, 0.0, None)
    f = jax.jit(jax.vmap(lambda x: mix(evaluate_points(*cfg.split(x), model.points), model.points.w)))
    return np.asarray(f(jnp.asarray(D)))


def plot_design(cfg: Config, d, out_dir: Path, history: dict | None = None) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, s in enumerate((1, 2)):
        lam, per, m = spectrum_vs_wavelength(cfg, d, s)
        ax = axes[0, col]
        ax.plot(lam, m[:, 1], label=f"T{s}")
        ax.plot(lam, m[:, 0], label=f"R{s}")
        ax.plot(lam, m[:, 2], label=f"A{s}", lw=0.8)
        ax.plot(lam, per[:, 0, 1], ":", lw=0.8, label=f"T{s} s")
        ax.plot(lam, per[:, 1, 1], "--", lw=0.8, label=f"T{s} p")
        ax.axvline(cfg.sources[s].wavelength_nm, color="k", lw=0.6)
        ax.set(xlabel="wavelength (nm)", ylabel="power fraction", ylim=(-0.02, 1.02),
               title=f"source {s} at {cfg.aoi_deg:g} deg ({cfg.sources[s].polarization})")
        ax.legend(fontsize=8)
        aoi, per, m = spectrum_vs_angle(cfg, d, s)
        ax = axes[1, col]
        ax.plot(aoi, m[:, 1], label=f"T{s}")
        ax.plot(aoi, m[:, 0], label=f"R{s}")
        ax.axvline(cfg.aoi_deg, color="k", lw=0.6)
        ax.set(xlabel="angle of incidence (deg)", ylabel="power fraction", ylim=(-0.02, 1.02),
               title=f"source {s} at {cfg.sources[s].wavelength_nm:g} nm")
        ax.legend(fontsize=8)
    p = out_dir / "spectra.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p)

    d_front, d_back = (np.asarray(a) for a in cfg.split(d))
    faces = [(f"front (source {cfg.plate.coated_face} side)", cfg.coating.materials, d_front),
             (f"back (source {3 - cfg.plate.coated_face} side)", cfg.back.materials, d_back)]
    fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
    colors = {m: f"C{i}" for i, m in enumerate(dict.fromkeys(cfg.coating.materials + cfg.back.materials))}
    for row, (_, mats, th) in enumerate(faces):
        x = 0.0
        for m, t in zip(mats, th):
            ax.barh(row, t, left=x, color=colors[m], edgecolor="k", lw=0.3)
            x += t
    for m, c in colors.items():
        ax.barh(0, 0, color=c, label=m)
    ax.set(xlabel="depth from ambient (nm)", yticks=[0, 1], yticklabels=[f[0] for f in faces],
           title="layer stacks (ambient -> substrate)")
    ax.legend(ncol=len(colors), fontsize=8, loc="upper right")
    p = out_dir / "stack.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p)

    if history and "adam_best_curve" in history:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.semilogy(history["adam_best_curve"])
        ax.set(xlabel="Adam step", ylabel="merit", title="convergence of the best start")
        p = out_dir / "convergence.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        files.append(p)
    return files
