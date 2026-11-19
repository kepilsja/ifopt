"""Write an optimisation run to a directory (the layout of results/<config>/ and the CLI output).

    res = fo.optimize(system)
    write_run("results/my_run", system, res)

Files: config.toml (the system), design.json (design + metadata), report.txt (evaluation report and
thickness Monte Carlo), spectra.png, stack.png, convergence.png.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from .api import OptimizationResult, monte_carlo
from .config import FilterSystem
from .plotting import save_plots

logger = logging.getLogger(__name__)


def _rel(p) -> str | None:
    if p is None:
        return None
    try:
        return os.path.relpath(p)
    except ValueError:
        return str(p)


def write_run(out_dir, system: FilterSystem, res: OptimizationResult, extra: dict | None = None,
              mc_samples: int = 2000) -> dict[str, Path]:
    """Write the run; returns {kind: path}.  ``extra`` is merged into design.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = res.evaluation.report() + "\n\n" + monte_carlo(system, res.design, n=mc_samples).summary()
    logger.info("")
    logger.info(text)
    paths = {"report": out / "report.txt", "config": out / "config.toml", "design": out / "design.json"}
    paths["report"].write_text(text + "\n")
    logger.info("saved %s", paths["report"])
    system.to_toml(paths["config"])
    logger.info("saved %s", paths["config"])
    res.design.save(paths["design"], system, config=_rel(system.source) or "config.toml", substrate_mode=system.mode,
                    merit=res.merit, mf=res.mf,
                    objective={"form": system.objective.form, "power": system.objective.power},
                    targets=[r.__dict__ for r in res.evaluation.targets], timings_s=res.timings,
                    top_candidates=[{"merit": v, "x_nm": d.x(system).tolist()} for v, d in res.candidates[:5]],
                    **(extra or {}))
    for p in save_plots(system, res.design, out, res.history):
        paths[p.stem] = p
    return paths
