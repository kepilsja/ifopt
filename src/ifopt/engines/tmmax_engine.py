"""Adapter that evaluates a single face (ambient | layers | substrate) with tmmax.

* Uses tmmax's jitted ``vectorized_coh_tmm`` with an nk ``data`` array built in memory from the
  complex indices we already have (one 2-point constant table per layer), so nothing is written
  into site-packages and no ``.npy`` files are read per call.
* tmmax hard-codes complex64 and fails when ``jax_enable_x64`` is on (scan carry complex64 vs
  complex128), so every call runs inside ``jax.enable_x64(False)``.
* Reflection/transmission from the substrate side are obtained by a second call with the
  layer order reversed and the incidence angle taken in the substrate.  tmmax accepts only a
  real angle in a real incident medium, so the reverse call uses Re(n_substrate); exact for
  lossless substrates (the case here).
* tmmax's incoherent-layer path is not used: see docs/DESIGN.md section 2 for why.
"""
from __future__ import annotations

import numpy as np

from .stable_engine import FacePowers


class TmmaxEngine:
    name = "tmmax (T-matrix, complex64)"

    def __init__(self):
        import jax
        from tmmax.tmm import vectorized_coh_tmm
        self._jax = jax
        self._f = vectorized_coh_tmm

    def _rt(self, n_list, d_nm, kx, lam_nm, pol: int):
        jax = self._jax
        jnp = jax.numpy
        n_list = np.asarray(n_list, dtype=complex)
        lam_m = lam_nm * 1e-9
        with jax.enable_x64(False):
            wl_nodes = np.array([lam_m * (1 - 1e-3), lam_m * (1 + 1e-3)])
            data = np.empty((n_list.size, 3, 2), dtype=np.float32)
            data[:, 0, :] = wl_nodes
            data[:, 1, :] = n_list.real[:, None]
            data[:, 2, :] = n_list.imag[:, None]
            theta0 = float(np.arcsin(kx / n_list[0].real))
            R, T = self._f(jnp.asarray(data), jnp.arange(n_list.size), jnp.asarray(np.asarray(d_nm, float) * 1e-9, jnp.float32),
                           jnp.asarray([lam_m], jnp.float32), jnp.asarray([theta0], jnp.float32), jnp.array([bool(pol)]))
            return float(np.asarray(R).ravel()[0]), float(np.asarray(T).ravel()[0])

    def face(self, n_in, n_layers, n_out, d_nm, kx, lam_nm, pol: int) -> FacePowers:
        n_layers = np.asarray(n_layers, dtype=complex)
        d_nm = np.asarray(d_nm, dtype=float)
        if n_layers.size == 0:
            # tmmax crashes on a bare interface (IndexError after jnp.squeeze); a zero-thickness
            # layer of the exit medium is optically identical.
            n_layers, d_nm = np.array([n_out], dtype=complex), np.array([0.0])
        fwd = np.concatenate([[n_in], n_layers, [n_out]])
        R, T = self._rt(fwd, d_nm, float(kx), float(lam_nm), pol)
        Rb, Tb = self._rt(fwd[::-1], d_nm[::-1], float(kx), float(lam_nm), pol)
        return FacePowers(R, T, Rb, Tb)
