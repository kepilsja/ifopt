"""Quick physics without targets or an optimiser: a hand-written quarter-wave mirror.

    .venv/bin/python examples/quick_stack.py
"""
import logging

import numpy as np

import ifopt as fo

logging.basicConfig(format="%(message)s")              # root stays at WARNING (no JAX/XLA chatter)
logging.getLogger("ifopt").setLevel(logging.INFO)       # ifopt progress messages
log = logging.getLogger("example")
log.setLevel(logging.INFO)

lib = fo.MaterialLibrary()
lam0 = 1064.0
layers = fo.alternating("TiO2", "FusedSilica", 17)
n = [lib.nk(m, lam0)[0] for m in layers]
d = [lam0 / (4 * ni.real) for ni in n]                     # quarter-wave thicknesses at normal incidence
r = fo.stack_rt(n, d, wavelength_nm=lam0, n_out=lib.nk("FusedSilica", lam0)[0])
log.info("17-layer QW mirror at %g nm, 0 deg: R = %.6f, T = %.2e", lam0, r.R, r.T)

# the same stack as a system (semi-infinite substrate) to sweep wavelength and angle
system = fo.FilterSystem(substrate=fo.Substrate("FusedSilica", mode="semi_infinite"),
                         front=fo.Coating(layers, thickness_nm=d),
                         targets=[fo.Target("HR", "R", 0.999, kind="min", wavelength_nm=lam0)])
design = fo.Design.from_system(system)
sp = fo.spectrum(system, design, wavelengths_nm=np.linspace(900, 1250, 8))
for lam, R in zip(sp.wavelengths_nm, sp.R):
    log.info("  %6.1f nm  R = %.4f", lam, R)
ang = fo.spectrum(system, design, wavelengths_nm=lam0, aoi_deg=[0, 20, 40, 60], polarization="both")
for a, (rs, rp) in zip(ang.aoi_deg, ang.per_pol[:, :, 0]):
    log.info("  AOI %4.0f deg  R_s = %.4f  R_p = %.4f", a, rs, rp)
