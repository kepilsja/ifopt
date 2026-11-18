"""Step 2: optimise the coating thicknesses to maximise a chosen pair of coefficients.

    python scripts/optimize_filter.py --config config/example.toml --maximize T1 R2
    python scripts/optimize_filter.py --config config/example.toml --port up     # = R1 + T2

Writes results/<run>/design.json, report.txt and plots (spectra, stack, convergence).
The final design is re-evaluated with the tmmax and reference engines as a cross-check.
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

from compute_rt import report  # noqa: E402
from ifopt.analysis import plot_design, thickness_monte_carlo  # noqa: E402
from ifopt.config import load_config, with_objective  # noqa: E402
from ifopt.geometry import PORTS  # noqa: E402
from ifopt.optimize import optimize  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/example.toml")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--maximize", nargs="+", metavar="COEF", help="e.g. T1 R2 (overrides config)")
    g.add_argument("--port", choices=sorted(PORTS), help="left = T1+R2, up = R1+T2")
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--starts", type=int, help="override optimizer.n_starts")
    ap.add_argument("--steps", type=int, help="override optimizer.adam_steps")
    ap.add_argument("--global-de", action="store_true", help="add a differential-evolution global search")
    ap.add_argument("--out", help="output directory (default results/<timestamp>)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.maximize or args.port:
        cfg = with_objective(cfg, tuple(args.maximize or PORTS[args.port]), tuple(args.weights) if args.weights else None)
    oc = cfg.optimizer
    cfg = replace(cfg, optimizer=replace(oc, n_starts=args.starts or oc.n_starts, adam_steps=args.steps or oc.adam_steps,
                                         global_de=args.global_de or oc.global_de))

    out = Path(args.out) if args.out else ROOT / "results" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Maximising {', '.join(cfg.objective.maximize)} ({cfg.objective.form}); {len(cfg.coating.materials)} front layers"
          + (f" + {len(cfg.back.materials)} back layers" if cfg.back_optimized else ""))
    res = optimize(cfg)

    text = report(cfg, res.d, f"optimised ({', '.join(cfg.objective.maximize)}), merit {res.merit:.3e}")
    mc = thickness_monte_carlo(cfg, res.d, rel_sigma=0.01)
    names = {"R1": (0, 0), "T1": (0, 1), "R2": (1, 0), "T2": (1, 1)}
    mc_lines = ["", "Thickness Monte Carlo (1% rel. sigma per layer, 2000 samples), mixed polarisation:"]
    for c in cfg.objective.maximize:
        s, k = names.get(c, (int(c[1]) - 1, 2))
        v = mc[:, s, k]
        mc_lines.append(f"  {c}: mean {v.mean():.4f}, 5th pct {np.percentile(v, 5):.4f}, worst {v.min():.4f}")
    text += "\n".join(mc_lines)
    print("\n" + text)

    (out / "report.txt").write_text(text + "\n")
    d_front, d_back = (np.asarray(a) for a in cfg.split(res.d))
    design = {"config": str(cfg.path), "maximize": list(cfg.objective.maximize), "weights": list(cfg.objective.weights),
              "merit": res.merit, "selected": res.selected, "materials": list(cfg.coating.materials),
              "thickness_nm": [float(x) for x in d_front],
              "back_materials": list(cfg.back.materials), "back_thickness_nm": [float(x) for x in d_back],
              "back_optimized": cfg.back_optimized, "timings_s": res.timings,
              "top_candidates": [{"merit": v, "thickness_nm": x} for v, x in res.candidates[:5]]}
    (out / "design.json").write_text(json.dumps(design, indent=2))
    files = plot_design(cfg, res.d, out, res.history)
    print(f"\nSaved {out / 'design.json'}, {out / 'report.txt'}, " + ", ".join(str(f) for f in files))


if __name__ == "__main__":
    main()
