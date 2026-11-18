"""Re-optimise the normal-incidence bands dichroic for several angles of incidence and substrate
models, and tabulate the results with pandas.

    .venv/bin/python examples/parameter_sweep.py
"""
import logging
from pathlib import Path

import pandas as pd

import ifopt as fo

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(format="%(message)s")              # root stays at WARNING (no JAX/XLA chatter)
logging.getLogger("ifopt").setLevel(logging.INFO)       # ifopt progress messages
log = logging.getLogger("example")
log.setLevel(logging.INFO)

base = fo.FilterSystem.from_toml(ROOT / "config" / "normal_dichroic_bands.toml")
settings = fo.OptimizerSettings(n_starts=32, adam_steps=300, top_k=3)

rows = []
for mode in ("semi_infinite", "incoherent"):
    for aoi in (0.0, 10.0, 20.0):
        targets = [fo.Target(t.name, t.quantity, t.value, kind=t.kind, wavelength_nm=t.wavelength_nm,
                             points=t.points, aoi_deg=aoi, polarization="both" if aoi else t.polarization,
                             side=t.side) for t in base.targets]
        system = base.replace(substrate=fo.Substrate("FusedSilica", mode=mode, thickness_mm=1.0), targets=targets)
        opt = fo.optimize(system, settings=settings)
        res = {t.name: t for t in opt.evaluation.targets}
        rows.append({"substrate": mode, "aoi_deg": aoi, "MF": opt.mf,
                     "pass T min": res["pass"].achieved_min, "reflect R min": res["reflect"].achieved_min,
                     "all met": all(t.met for t in opt.evaluation.targets)})
        log.info("%-13s AOI %4.1f deg: MF %.3e", mode, aoi, opt.mf)

table = pd.DataFrame(rows)
log.info("\n%s", table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
