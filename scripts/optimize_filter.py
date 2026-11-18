"""Optimise the layer thicknesses against the target table of a config.

    python scripts/optimize_filter.py --config config/combiner_45.toml [--out results/combiner_45]

Writes design.json, report.txt and plots (spectra per beam, stack, convergence) to the output
directory (default results/<timestamp>).  The final design is cross-checked with the tmmax and
reference engines in the report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compute_rt import rel, report  # noqa: E402
from ifopt.analysis import plot_design, thickness_monte_carlo  # noqa: E402
from ifopt.config import load_config  # noqa: E402
from ifopt.objectives import figure_of_merit  # noqa: E402
from ifopt.optimize import optimize  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "combiner_45.toml"))
    ap.add_argument("--starts", type=int, help="override optimizer.n_starts")
    ap.add_argument("--steps", type=int, help="override optimizer.adam_steps")
    ap.add_argument("--global-de", action="store_true", help="add a differential-evolution global search")
    ap.add_argument("--out", help="output directory (default results/<timestamp>)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    oc = cfg.optimizer
    cfg = replace(cfg, optimizer=replace(oc, n_starts=args.starts or oc.n_starts, adam_steps=args.steps or oc.adam_steps,
                                         global_de=args.global_de or oc.global_de))
    out = Path(args.out) if args.out else ROOT / "results" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    t = cfg.table
    print(f"{len(t.rows)} target rows -> {t.t_point.size} target points at {t.n_points} evaluation points; "
          f"{cfg.n_front} front layers" + (f" + {len(cfg.back.materials)} back layers" if cfg.back_optimized else "")
          + f"; substrate mode {cfg.mode}")
    res = optimize(cfg)

    text = report(cfg, res.d, f"optimised, merit {res.merit:.3e}")
    m, worst = thickness_monte_carlo(cfg, res.d, rel_sigma=0.01)
    lines = ["", "Thickness Monte Carlo (1% rel. sigma per layer, 2000 samples):",
             f"  MF: nominal {res.mf:.4e}, median {np.median([figure_of_merit(v, cfg.objective) for v in m]):.4e}, "
             f"95th pct {figure_of_merit(np.percentile(m, 95), cfg.objective):.4e}"]
    for i, r in enumerate(t.rows):
        w = worst[:, i]
        met = f", requirement met in {100 * np.mean(w <= 1e-12):.0f}% of samples" if r.kind != "eq" else ""
        lines.append(f"  {r.name}: worst |d| median {np.median(w):.2e}, 95th pct {np.percentile(w, 95):.2e}{met}")
    text += "\n" + "\n".join(lines)
    print("\n" + text)
    (out / "report.txt").write_text(text + "\n")

    d_front, d_back = (np.asarray(a) for a in cfg.split(res.d))
    design = {"config": rel(cfg.path), "substrate_mode": cfg.mode, "merit": res.merit, "mf": res.mf,
              "objective": {"form": cfg.objective.form, "power": cfg.objective.power},
              "materials": list(cfg.front.materials), "thickness_nm": [float(v) for v in d_front],
              "back_materials": list(cfg.back.materials), "back_thickness_nm": [float(v) for v in d_back],
              "back_optimized": cfg.back_optimized, "targets": res.rows, "timings_s": res.timings,
              "top_candidates": [{"merit": v, "x_nm": xx} for v, xx in res.candidates[:5]]}
    (out / "design.json").write_text(json.dumps(design, indent=2))
    files = plot_design(cfg, res.d, out, res.history)
    print(f"\nSaved {rel(out / 'design.json')}, {rel(out / 'report.txt')}, " + ", ".join(rel(f) for f in files))


if __name__ == "__main__":
    main()
