"""Build the 45-degree beam combiner in code, check it equals the TOML config, evaluate, optimise, save.

    .venv/bin/python examples/scripted_combiner.py
"""
import logging
from pathlib import Path

import ifopt as fo

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(format="%(message)s")              # root stays at WARNING (no JAX/XLA chatter)
logging.getLogger("ifopt").setLevel(logging.INFO)       # ifopt progress messages
log = logging.getLogger("example")
log.setLevel(logging.INFO)

# operating conditions: source 1 on the front (coated) face, source 2 on the back face (system.xml)
b1 = fo.Beam("1", wavelength_nm=532, aoi_deg=45, side="front", polarization="s")
b2 = fo.Beam("2", wavelength_nm=1064, aoi_deg=45, side="back", polarization="s")

system = fo.FilterSystem(
    substrate=fo.Substrate("FusedSilica", mode="incoherent", thickness_mm=1.0),
    front=fo.Coating(fo.alternating("TiO2", "FusedSilica", 15), bounds_nm={"default": (5, 400)}),
    targets=[b1.eq("T", 1.0, name="T1"), b2.eq("R", 1.0, name="R2")],
    sums=[fo.ReportSum("left port", ["T@1", "R@2"]), fo.ReportSum("up port", ["R@1", "T@2"])],
)
same = system == fo.FilterSystem.from_toml(ROOT / "config" / "combiner_45.toml")
log.info("code-built system equals config/combiner_45.toml: %s", same)

seed = fo.Design.quarter_wave(system)
log.info("quarter-wave seed: T1 = %.4f, R2 = %.4f", fo.evaluate(system, seed).beams["1"].T[0],
         fo.evaluate(system, seed).beams["2"].R[0])

opt = fo.optimize(system, settings=fo.OptimizerSettings(n_starts=64, adam_steps=400))
ev = opt.evaluation
log.info("optimised:  T1 = %.6f (back-face ceiling %.6f), R2 = %.6f, left port = %.6f",
         ev.beams["1"].T[0], ev.ceiling["1"], ev.beams["2"].R[0], ev.sums["left port"])

# a variant: the same targets with an optimised 4-layer back-face AR coating
with_ar = system.replace(back=fo.Coating(["FusedSilica", "TiO2", "FusedSilica", "TiO2"], bounds_nm={"default": (5, 300)}))
opt_ar = fo.optimize(with_ar, settings=fo.OptimizerSettings(n_starts=64, adam_steps=400))
log.info("with back-face AR: T1 = %.6f, R2 = %.6f", opt_ar.evaluation.beams["1"].T[0], opt_ar.evaluation.beams["2"].R[0])

out = ROOT / "results" / "examples"
out.mkdir(parents=True, exist_ok=True)
opt.design.save(out / "combiner_design.json", system)
fo.plot_spectra(system, opt.design).savefig(out / "combiner_spectra.png", dpi=110)
log.info("saved %s", out)
