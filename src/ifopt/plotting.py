"""Figures (matplotlib ``Figure`` objects, no pyplot state or backend changes).

    fig = plot_spectra(system, design); fig.savefig("spectra.png")

In Jupyter the returned figures display inline.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .config import FilterSystem
from .design import as_x

logger = logging.getLogger(__name__)
COLORS = {"R": "C1", "T": "C0", "A": "C2"}
MARK = {"eq": "o", "min": "^", "max": "v"}
REL = {"eq": "=", "min": ">=", "max": "<="}


def _figure(figsize):
    from matplotlib.figure import Figure
    return Figure(figsize=figsize, layout="constrained")


def plot_spectra(system: FilterSystem, design, beams=None):
    """Per beam: R/T/A vs wavelength with the target requirements (top), R/T vs AOI (bottom)."""
    from .analysis import beam_spectrum, beam_vs_angle
    x = as_x(system, design)
    beams = list(beams or system.table.beams)
    fig = _figure((6 * len(beams), 8))
    axes = fig.subplots(2, len(beams), squeeze=False)
    for col, label in enumerate(beams):
        b = system.table.beams[label]
        lam, per, m = beam_spectrum(system, x, label)
        ax = axes[0, col]
        for k, q in enumerate("RTA"):
            ax.plot(lam, m[:, k], color=COLORS[q], lw=1.6 if q != "A" else 0.9, label=q)
        if system.has_back_face or b.aois_deg != (0.0,):
            ax.plot(lam, per[:, 0, 1], ":", color="C3", lw=0.8, label="T s")
            ax.plot(lam, per[:, 1, 1], "--", color="C4", lw=0.8, label="T p")
        for i in b.rows:
            r = system.targets[i]
            lo, hi = min(r.wavelengths_nm), max(r.wavelengths_nm)
            if hi > lo:
                ax.hlines(r.value, lo, hi, color=COLORS[r.quantity], lw=3, alpha=0.35)
            ax.plot([0.5 * (lo + hi)], [r.value], MARK[r.kind], color=COLORS[r.quantity], ms=8, mec="k",
                    label=f"{r.name}: {r.quantity} {REL[r.kind]} {r.value:g}")
        ax.set(xlabel="wavelength (nm)", ylabel="power fraction", ylim=(-0.02, 1.02),
               title=f"beam {label}: {b.side} side, AOI {0.5 * (min(b.aois_deg) + max(b.aois_deg)):g} deg, "
                     f"pol {b.polarization}")
        ax.legend(fontsize=7)
        aoi, per, m = beam_vs_angle(system, x, label)
        ax = axes[1, col]
        for k, q in enumerate("RT"):
            ax.plot(aoi, m[:, k], color=COLORS[q], label=q)
        ax.set(xlabel="angle of incidence (deg)", ylabel="power fraction", ylim=(-0.02, 1.02),
               title=f"beam {label} at {0.5 * (min(b.wavelengths_nm) + max(b.wavelengths_nm)):g} nm")
        ax.legend(fontsize=7)
    return fig


def plot_stack(system: FilterSystem, design, ax=None):
    """Layer thickness bars for the front (and back) coating."""
    d_front, d_back = (np.asarray(a) for a in system.split(as_x(system, design)))
    faces = [("front", system.front.materials, d_front)]
    if system.back.materials:
        faces.append(("back", system.back.materials, d_back))
    fig = None
    if ax is None:
        fig = _figure((9, 1.5 + 1.2 * len(faces)))
        ax = fig.subplots()
    colors = {m: f"C{i}" for i, m in enumerate(dict.fromkeys(system.front.materials + system.back.materials))}
    for row, (_, mats, th) in enumerate(faces):
        pos = 0.0
        for m_, t_ in zip(mats, th):
            ax.barh(row, t_, left=pos, color=colors[m_], edgecolor="k", lw=0.3)
            pos += t_
    for m_, c in colors.items():
        ax.barh(0, 0, color=c, label=m_)
    ax.set(xlabel="depth from ambient (nm)", yticks=list(range(len(faces))), yticklabels=[f[0] for f in faces],
           title=f"layer stacks (ambient -> substrate), substrate mode: {system.mode}")
    ax.legend(ncol=len(colors), fontsize=8, loc="upper right")
    return fig if fig is not None else ax.figure


def plot_convergence(history: dict, ax=None):
    """Merit of the best Adam start vs step (zeros are not drawn on the log axis)."""
    fig = None
    if ax is None:
        fig = _figure((6, 4))
        ax = fig.subplots()
    curve = np.asarray(history["adam_best_curve"], float)
    ax.semilogy(np.where(curve > 0, curve, np.nan))
    ax.set(xlabel="Adam step", ylabel="merit", title="convergence of the best start")
    return fig if fig is not None else ax.figure


def save_plots(system: FilterSystem, design, out_dir, history: dict | None = None, dpi: int = 120) -> list[Path]:
    """Write spectra.png, stack.png (and convergence.png) to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = [("spectra.png", plot_spectra(system, design)), ("stack.png", plot_stack(system, design))]
    if history and "adam_best_curve" in history:
        figs.append(("convergence.png", plot_convergence(history)))
    files = []
    for name, fig in figs:
        path = out_dir / name
        fig.savefig(path, dpi=dpi)
        files.append(path)
        logger.info("saved %s", path)
    return files
