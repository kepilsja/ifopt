"""Step 1: transmittance / reflectance / absorptance of the plate for both sources.

    python scripts/compute_rt.py --config config/example.toml [--design results/<run>/design.json]

Prints R, T, A per source for s, p and the configured polarisation, computed by
  * stable    - own S-matrix engine, complex128 (default, used by the optimiser)
  * tmmax     - tmmax coherent engine (complex64) + same incoherent substrate formula
  * reference - Byrnes' tmm package (float64)
plus the coherent-substrate fringe envelope and numerical-stability diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ifopt.config import load_config  # noqa: E402
from ifopt.engines import ReferenceEngine, TmmaxEngine  # noqa: E402
from ifopt.geometry import OUTPUT_PORT, SOURCE_FACE  # noqa: E402
from ifopt.objectives import quarter_wave_seed  # noqa: E402
from ifopt.plate import PlateModel, evaluate_with_engine  # noqa: E402

F32_LIMIT, F64_LIMIT = 88.72, 709.78


def design_from(cfg, design_path: str | None):
    if design_path:
        d = json.loads(Path(design_path).read_text())
        if list(d["materials"]) != list(cfg.coating.materials):
            raise SystemExit("design materials do not match the config's coating.materials")
        parts = [np.asarray(d["thickness_nm"], float)]
        if cfg.back_optimized:
            if list(d.get("back_materials", [])) != list(cfg.back.materials):
                raise SystemExit("design back_materials do not match the config's back_coating.materials")
            parts.append(np.asarray(d["back_thickness_nm"], float))
        return np.concatenate(parts), f"design file {design_path}"
    if (x := cfg.configured_design()) is not None:
        return x, "thicknesses from config"
    return quarter_wave_seed(cfg), "quarter-wave seed (no thicknesses given)"


def report(cfg, d, origin: str, engines=("tmmax", "reference")) -> str:
    model = PlateModel(cfg)
    lines = []
    out = lines.append
    out(f"Config: {cfg.path}")
    out(f"Design: {origin}")
    out(f"AOI {cfg.aoi_deg:g} deg, ambient {cfg.ambient}, substrate {cfg.plate.substrate} "
        f"{cfg.plate.substrate_thickness_mm:g} mm, coating on the face of source {cfg.plate.coated_face}, "
        f"back face: {'bare' if not cfg.back.materials else ('optimised' if cfg.back_optimized else 'fixed') + ' coating'}")
    d_front, d_back = (np.asarray(a) for a in cfg.split(d))
    out("Front layers (ambient -> substrate): " + ", ".join(f"{m} {t:.2f}" for m, t in zip(cfg.coating.materials, d_front)) + " nm")
    if cfg.back.materials:
        out("Back layers  (ambient -> substrate): " + ", ".join(f"{m} {t:.2f}" for m, t in zip(cfg.back.materials, d_back)) + " nm")
    for s in (1, 2):
        sc = cfg.sources[s]
        out(f"Source {s}: {sc.wavelength_nm:g} nm, polarization {sc.polarization!r}, enters the {SOURCE_FACE[s]} "
            f"({'coated' if cfg.plate.coated_face == s else 'back'} face)")
    out("")
    stable = model.coefficients(d)
    others = {}
    for name in engines:
        try:
            eng = TmmaxEngine() if name == "tmmax" else ReferenceEngine()
            others[name] = evaluate_with_engine(cfg, d, eng)
        except Exception as exc:  # keep the report going
            out(f"[{name} engine failed: {type(exc).__name__}: {exc}]")
    lo, mean, hi = model.envelope(d)
    hdr = f"{'coef':5s} {'port':5s} {'s':>10s} {'p':>10s} {'mixed':>10s}"
    for name in others:
        hdr += f" {'|d ' + name + '|':>14s}"
    hdr += f" {'coherent min':>13s} {'max':>8s}"
    out(hdr)
    for s in (1, 2):
        for k, q in enumerate("RTA"):
            c = f"{q}{s}"
            v = {x: (0.0 if abs(y) < 5e-13 else y) for x, y in stable[c].items()}
            row = f"{c:5s} {OUTPUT_PORT.get(c, '-'):5s} {v['s']:10.6f} {v['p']:10.6f} {v['mixed']:10.6f}"
            for name, o in others.items():
                diff = max(abs(o[c][x] - v[x]) for x in ("s", "p", "mixed"))
                row += f" {diff:14.2e}"
            row += f" {lo[s - 1, k] + 0.0:13.6f} {hi[s - 1, k] + 0.0:8.6f}"
            out(row)
    out("")
    left = stable["T1"]["mixed"] + stable["R2"]["mixed"]
    up = stable["R1"]["mixed"] + stable["T2"]["mixed"]
    out(f"Port totals (mixed pol., each beam normalised to its own source, max 2): "
        f"left = T1 + R2 = {left:.6f}   up = R1 + T2 = {up:.6f}")
    ceil = model.back_face_ceiling(d)
    out(f"Ceiling set by the {'bare' if not cfg.back.materials else 'current'} back face (any passive front coating, "
        f"incoherent substrate): T1 <= {ceil[0]:.6f}, T2 <= {ceil[1]:.6f}")
    sid = model.sum_im_delta(d)
    out(f"Stability: sum Im(delta) in coating = {sid[0]:.3g} (source 1), {sid[1]:.3g} (source 2); "
        f"T-matrix overflow at ~{F32_LIMIT} (complex64, tmmax) / ~{F64_LIMIT} (complex128); "
        f"S-matrix engine has no limit.")
    out("Coherent min/max: substrate treated as an etalon (single-frequency laser, parallel plate); "
        "its phase average equals the incoherent value.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/example.toml")
    ap.add_argument("--design", help="JSON written by optimize_filter.py")
    ap.add_argument("--engines", default="tmmax,reference", help="comparison engines (comma list, may be empty)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    d, origin = design_from(cfg, args.design)
    print(report(cfg, d, origin, tuple(e for e in args.engines.split(",") if e)))


if __name__ == "__main__":
    main()
