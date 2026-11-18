"""Evaluate a design against the target table: R / T / A per beam, targets, merit.

    python scripts/compute_rt.py --config config/combiner_45.toml [--design results/<run>/design.json]

Per beam: R, T, A for s, p and the beam's polarisation, computed by
  * stable    - own S-matrix engine, complex128 (default, used by the optimiser)
  * tmmax     - tmmax coherent engine (complex64) + the same substrate model   (difference shown)
  * reference - Byrnes' tmm package (float64)                                   (difference shown)
plus, in the plate modes, the coherent-substrate fringe envelope and the back-face ceiling.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ifopt.config import load_config  # noqa: E402
from ifopt.engines import ReferenceEngine, TmmaxEngine  # noqa: E402
from ifopt.objectives import figure_of_merit, make_merit, quarter_wave_seed, row_summary  # noqa: E402
from ifopt.plate import Model, evaluate_with_engine  # noqa: E402

F32_LIMIT, F64_LIMIT = 88.72, 709.78
SYM = {"eq": "=", "min": ">=", "max": "<="}


def z(v) -> float:
    """Print rounding residue such as -1e-17 as 0."""
    return 0.0 if abs(v) < 5e-13 else float(v)


def rel(p) -> str:
    try:
        return os.path.relpath(p, ROOT)
    except ValueError:
        return str(p)


def design_from(cfg, design_path: str | None):
    if design_path:
        d = json.loads(Path(design_path).read_text())
        if list(d["materials"]) != list(cfg.front.materials):
            raise SystemExit("design materials do not match the config's [front] materials")
        parts = [np.asarray(d["thickness_nm"], float)]
        if cfg.back_optimized:
            if list(d.get("back_materials", [])) != list(cfg.back.materials):
                raise SystemExit("design back_materials do not match the config's [back] materials")
            parts.append(np.asarray(d["back_thickness_nm"], float))
        return np.concatenate(parts), f"design file {rel(design_path)}"
    if (x := cfg.configured_design()) is not None:
        return x, "thicknesses from config"
    return quarter_wave_seed(cfg), "quarter-wave seed (no thicknesses given)"


def _span(vals, unit):
    lo, hi = min(vals), max(vals)
    return f"{lo:g} {unit}" if lo == hi else f"{lo:g}-{hi:g} {unit} ({len(vals)} pts)"


def report(cfg, x, origin: str, engines=("tmmax", "reference"), max_engine_points: int = 40) -> str:
    model = Model(cfg)
    t = cfg.table
    res = np.asarray(model.response(x))
    L = []
    out = L.append
    out(f"Config: {rel(cfg.path)}")
    out(f"Design: {origin}")
    sub = cfg.substrate
    desc = {"incoherent": f"{sub.material} {sub.thickness_mm:g} mm, incoherent",
            "coherent": f"{sub.material} {sub.thickness_mm:g} mm, coherent (etalon)",
            "semi_infinite": f"{sub.material}, semi-infinite (back surface ignored)",
            "none": f"none - exit medium {cfg.exit_material}"}[cfg.mode]
    out(f"Ambient {cfg.ambient}; substrate: {desc}")
    d_front, d_back = (np.asarray(a) for a in cfg.split(x))
    out("Front layers (ambient -> substrate): " + ", ".join(f"{m} {v:.2f}" for m, v in zip(cfg.front.materials, d_front)) + " nm")
    if cfg.back.materials:
        out(f"Back layers  (ambient -> substrate, {'optimised' if cfg.back_optimized else 'fixed'}): "
            + ", ".join(f"{m} {v:.2f}" for m, v in zip(cfg.back.materials, d_back)) + " nm")
    elif cfg.has_back_face:
        out("Back face: bare")

    sel = np.arange(t.n_points) if t.n_points <= max_engine_points else \
        np.unique(np.linspace(0, t.n_points - 1, max_engine_points).round().astype(int))
    diffs = {}
    for name in engines:
        try:
            eng = TmmaxEngine() if name == "tmmax" else ReferenceEngine()
            full = np.full((t.n_points, 2, 3), np.nan)
            full[sel] = np.abs(evaluate_with_engine(cfg, x, eng, sel) - res[sel])
            diffs[name] = full
        except Exception as exc:
            out(f"[{name} cross-check unavailable: {type(exc).__name__}: {exc}]")
    scan = model.coherent_scan(x)
    ceil = model.back_face_ceiling(x)

    out("")
    for label, b in t.beams.items():
        w = np.asarray(b.weights)
        per, mixed = model.beam_response(x, label, res)
        out(f"Beam {label}: {_span(b.wavelengths_nm, 'nm')}, AOI {_span(b.aois_deg, 'deg')}, {b.side} side, "
            f"polarization {b.polarization!r}")
        eng_cols = "".join(f" {'|d ' + n + '|':>13s}" for n in diffs)
        if b.single:
            p = int(b.points[0])
            env = None if scan is None else np.einsum("p,npk->nk", w, scan[p])
            out(f"  {'':3s} {'s':>10s} {'p':>10s} {'beam':>10s}{eng_cols}" + ("   coherent min      max" if env is not None else ""))
            for k, q in enumerate("RTA"):
                row = f"  {q:3s} {z(per[0, 0, k]):10.6f} {z(per[0, 1, k]):10.6f} {z(mixed[0, k]):10.6f}"
                row += "".join(f" {np.nanmax(dd[p, :, k]):13.2e}" if np.isfinite(dd[p]).any() else f" {'-':>13s}"
                               for dd in diffs.values())
                if env is not None:
                    row += f"   {z(env[:, k].min()):12.6f} {z(env[:, k].max()):8.6f}"
                out(row)
            if ceil is not None:
                out(f"  back-face ceiling (any passive front coating): T <= {float(w @ ceil[p]):.6f}")
        else:
            out(f"  {'':3s} {'min':>10s} {'mean':>10s} {'max':>10s}{eng_cols}   (beam polarization)")
            for k, q in enumerate("RTA"):
                row = f"  {q:3s} {z(mixed[:, k].min()):10.6f} {z(mixed[:, k].mean()):10.6f} {z(mixed[:, k].max()):10.6f}"
                for dd in diffs.values():
                    v = dd[b.points][:, :, k]
                    row += f" {np.nanmax(v):13.2e}" if np.isfinite(v).any() else f" {'-':>13s}"
                out(row)
        out("")

    merit, _ = make_merit(model)
    mv = float(merit(x))
    ob = cfg.objective
    out(f"{'target':12s} {'beam':8s} {'requirement':16s} {'achieved (min..max)':>24s} {'worst |d|':>10s} {'share':>7s}")
    for r in row_summary(model, x):
        ach = f"{r['achieved_min']:.6f}" if r["points"] == 1 else f"{r['achieved_min']:.6f}..{r['achieved_max']:.6f}"
        out(f"{r['name']:12s} {r['beam']:8s} {r['quantity'] + ' ' + SYM[r['kind']] + ' ' + format(r['value'], 'g'):16s} "
            f"{ach:>24s} {r['worst_deviation']:10.2e} {100 * r['share']:6.1f}%")
    form = f"power, p = {ob.power:g}" if ob.form == "power" else ob.form
    out(f"Merit ({form}): {mv:.4e}" + (f"   MF = {figure_of_merit(mv, ob):.4e}" if ob.form == "power" else ""))
    for s in cfg.report_sums:
        total = 0.0
        for term in s.terms:
            q, beam = t.parse_term(term)
            total += float(model.beam_response(x, beam, res)[1][0, "RTA".index(q)])
        out(f"{s.name} = {' + '.join(s.terms)} = {total:.6f}")
    sid = model.sum_im_delta(x)
    out(f"Stability: max sum Im(delta) over points = {sid.max():.3g} (T-matrix overflow at ~{F32_LIMIT} in complex64 / "
        f"~{F64_LIMIT} in complex128; the S-matrix engine has no limit)")
    if scan is not None:
        out("Coherent min/max: substrate as an etalon (single-frequency laser, parallel plate); "
            "its phase average equals the incoherent value.")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "combiner_45.toml"))
    ap.add_argument("--design", help="JSON written by optimize_filter.py")
    ap.add_argument("--engines", default="tmmax,reference", help="comparison engines (comma list, may be empty)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    x, origin = design_from(cfg, args.design)
    print(report(cfg, x, origin, tuple(e for e in args.engines.split(",") if e)))


if __name__ == "__main__":
    main()
