"""Command line interface (``ifopt`` / ``python -m ifopt``), a thin layer over the public API.

    ifopt evaluate  --config config/combiner_45.toml [--design results/combiner_45/design.json] [--engines tmmax,reference]
    ifopt optimize  --config config/normal_dichroic_bands.toml [--out DIR] [--starts N] [--steps N] [--global-de]
    ifopt stability [--config CFG | --wavelengths 532 1064] [--out results/stability]

All output goes through the ``ifopt`` loggers: -v (DEBUG), -q (WARNING only), --log-file PATH.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

logger = logging.getLogger(__name__)
_HANDLER_TAG = "_ifopt_cli"


def _configure_logging(verbose: int, quiet: bool, log_file: str | None) -> None:
    root = logging.getLogger("ifopt")
    for h in list(root.handlers):                          # repeated main() calls (tests, notebooks)
        if getattr(h, _HANDLER_TAG, False):
            root.removeHandler(h)
            h.close()
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    console.setLevel(level)
    setattr(console, _HANDLER_TAG, True)
    root.addHandler(console)
    root.setLevel(logging.DEBUG if (verbose or log_file) else level)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        fh.setLevel(logging.DEBUG)
        setattr(fh, _HANDLER_TAG, True)
        root.addHandler(fh)


def _rel(p) -> str:
    try:
        return os.path.relpath(p)
    except ValueError:
        return str(p)


def _cmd_evaluate(args) -> int:
    from .api import evaluate
    from .config import FilterSystem
    from .design import Design
    system = FilterSystem.from_toml(args.config)
    if args.design:
        design, origin = Design.load(args.design, system), f"design file {_rel(args.design)}"
    elif system.configured_design() is not None:
        design, origin = Design.from_system(system), "thicknesses from config"
    else:
        design, origin = Design.quarter_wave(system), "quarter-wave seed (no thicknesses given)"
    engines = tuple(e for e in args.engines.split(",") if e)
    logger.info(evaluate(system, design, engines=engines, label=origin).report())
    return 0


def _cmd_optimize(args) -> int:
    from .api import monte_carlo, optimize
    from .config import FilterSystem
    from .plotting import save_plots
    system = FilterSystem.from_toml(args.config)
    oc = system.optimizer
    settings = replace(oc, n_starts=args.starts or oc.n_starts, adam_steps=args.steps or oc.adam_steps,
                       global_de=args.global_de or oc.global_de)
    out = Path(args.out) if args.out else Path("results") / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    t = system.table
    logger.info("%d targets -> %d target points at %d evaluation points; %d front layers%s; substrate mode %s",
                len(t.rows), t.t_point.size, t.n_points, system.n_front,
                f" + {len(system.back.materials)} back layers" if system.back_optimized else "", system.mode)
    res = optimize(system, settings=settings, engines=("tmmax", "reference"))
    text = res.evaluation.report() + "\n\n" + monte_carlo(system, res.design).summary()
    logger.info("")
    logger.info(text)
    (out / "report.txt").write_text(text + "\n")
    res.design.save(out / "design.json", system, config=_rel(system.source), substrate_mode=system.mode,
                    merit=res.merit, mf=res.mf,
                    objective={"form": system.objective.form, "power": system.objective.power},
                    targets=[r.__dict__ for r in res.evaluation.targets], timings_s=res.timings,
                    top_candidates=[{"merit": v, "x_nm": d.x(system).tolist()} for v, d in res.candidates[:5]])
    logger.info("saved %s", out / "report.txt")
    save_plots(system, res.design, out, res.history)
    return 0


def _cmd_stability(args) -> int:
    from .stability import run
    logger.info(run(args.out, config=args.config, wavelengths=args.wavelengths))
    return 0


def _logging_options(suppress: bool) -> argparse.ArgumentParser:
    """Logging options, accepted before or after the subcommand (the subcommand copy must not
    overwrite a value given before it, hence SUPPRESS defaults there)."""
    p = argparse.ArgumentParser(add_help=False)
    kw = {"default": argparse.SUPPRESS} if suppress else {}
    p.add_argument("-v", "--verbose", action="count", help="debug output", **(kw or {"default": 0}))
    p.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only", **kw)
    p.add_argument("--log-file", help="also log (with timestamps, DEBUG level) to this file", **kw)
    return p


def build_parser() -> argparse.ArgumentParser:
    common = _logging_options(suppress=True)
    ap = argparse.ArgumentParser(prog="ifopt", description="Interference filter evaluation and optimisation",
                                 parents=[_logging_options(suppress=False)])
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("evaluate", parents=[common], help="evaluate a design against the targets of a config")
    e.add_argument("--config", required=True)
    e.add_argument("--design", help="design.json written by 'ifopt optimize'")
    e.add_argument("--engines", default="tmmax,reference", help="comparison engines (comma list, may be empty)")
    e.set_defaults(func=_cmd_evaluate)
    o = sub.add_parser("optimize", parents=[common], help="optimise layer thicknesses against the targets")
    o.add_argument("--config", required=True)
    o.add_argument("--starts", type=int, help="override optimizer.n_starts")
    o.add_argument("--steps", type=int, help="override optimizer.adam_steps")
    o.add_argument("--global-de", action="store_true", help="add a differential-evolution global search")
    o.add_argument("--out", help="output directory (default results/<timestamp>)")
    o.set_defaults(func=_cmd_optimize)
    s = sub.add_parser("stability", parents=[common], help="numerical-stability study of tmmax vs the S-matrix engine")
    s.add_argument("--config", help="audit the nk database at this config's beam wavelengths")
    s.add_argument("--wavelengths", nargs="+", type=float, help="audit wavelengths in nm (default 532 1064)")
    s.add_argument("--out", default="results/stability")
    s.set_defaults(func=_cmd_stability)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose, args.quiet, args.log_file)
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        logger.error("error: %s", exc)
        return 2
