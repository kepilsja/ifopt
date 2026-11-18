"""ifopt - interference filter optimisation for a two-source 45-degree dichroic plate.

Importing the package switches JAX to 64-bit mode: the stable engine and the optimiser
work in float64/complex128.  tmmax is incompatible with x64 mode (its scan carry is
hard-coded complex64), so the tmmax adapter runs inside ``jax.enable_x64(False)``.
"""
import jax

jax.config.update("jax_enable_x64", True)

__all__ = ["config", "geometry", "materials", "plate", "objectives", "optimize", "analysis"]
