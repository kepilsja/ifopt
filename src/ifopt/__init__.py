"""ifopt - interference filter design: target tables, a stable S-matrix engine and optimisation.

Scripting::

    import ifopt as fo
    system = fo.FilterSystem.from_toml("config/combiner_45.toml")      # or build it in code
    ev = fo.evaluate(system, fo.Design.quarter_wave(system))
    opt = fo.optimize(system)

Importing the package switches JAX to 64-bit mode (the S-matrix engine and the optimiser work in
float64/complex128); public API functions also force 64-bit mode while they run.  tmmax is
incompatible with x64 mode, so the tmmax adapter runs inside ``jax.enable_x64(False)``.

The package logs through ``logging.getLogger("ifopt")`` and never prints; call
``enable_logging()`` (or configure logging yourself) to see progress messages.
"""
import logging as _logging

import jax as _jax

_jax.config.update("jax_enable_x64", True)
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__version__ = "0.2.0"

from .api import (BeamResult, Evaluation, MonteCarlo, OptimizationResult, Spectrum, StackRT,  # noqa: E402
                  TargetResult, angle_scan, beam_spectrum, evaluate, monte_carlo, optimize, report, spectrum,
                  stack_rt)
from .config import (Coating, FilterSystem, Objective, OptimizerSettings, ReportSum, Substrate,  # noqa: E402
                     load_config)
from .design import Design  # noqa: E402
from .materials import (ConstantMaterial, Material, MaterialLibrary, MaterialRangeError,  # noqa: E402
                        SellmeierMaterial, TabulatedMaterial)
from .plotting import plot_convergence, plot_spectra, plot_stack, save_plots  # noqa: E402
from .targets import Beam, Target, alternating  # noqa: E402


def enable_logging(level="INFO", stream=None, fmt: str = "%(message)s") -> _logging.Handler:
    """Show ifopt log messages (e.g. optimiser progress) on ``stream`` (default stderr).
    Attaches one handler to the ``ifopt`` logger only; calling it again replaces that handler."""
    log = _logging.getLogger(__name__)
    for h in list(log.handlers):
        if getattr(h, "_ifopt_enable_logging", False):
            log.removeHandler(h)
    handler = _logging.StreamHandler(stream)
    handler.setFormatter(_logging.Formatter(fmt))
    handler._ifopt_enable_logging = True
    log.addHandler(handler)
    log.setLevel(level)
    return handler


__all__ = [
    "Beam", "BeamResult", "Coating", "ConstantMaterial", "Design", "Evaluation", "FilterSystem", "Material",
    "MaterialLibrary", "MaterialRangeError", "MonteCarlo", "Objective", "OptimizationResult", "OptimizerSettings",
    "ReportSum", "SellmeierMaterial", "Spectrum", "StackRT", "Substrate", "TabulatedMaterial", "Target",
    "TargetResult", "alternating", "angle_scan", "beam_spectrum", "enable_logging", "evaluate", "load_config",
    "monte_carlo", "optimize", "plot_convergence", "plot_spectra", "plot_stack", "report", "save_plots", "spectrum",
    "stack_rt", "__version__",
]
