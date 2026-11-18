"""Unconditionally stable thin-film solver: scattering matrices + Redheffer star product (JAX, complex128).

Why not the transfer matrix used by tmmax/Byrnes?  The T-matrix propagates a layer with
diag(e^{-i delta}, e^{+i delta}); in an absorbing layer (Im delta > 0) the first entry grows like
e^{Im delta}, so the cascade overflows (float32: sum Im delta ~ 88.7, float64: ~ 709.8) and the
reverse-mode gradient, which forms 1/M00^2, breaks at roughly half that.  The S-matrix only
ever uses the *decaying* factor e^{+i delta} (|.| <= 1) and the denominators 1 - r'_A r_B, which
are bounded away from zero for passive media, so any thickness/loss is handled.

Conventions follow Byrnes (arXiv:1603.02720), as do tmmax and the ``tmm`` package, so all
three engines agree term by term:
  q_j = n_j cos(theta_j) = sqrt(n_j^2 - kx^2), kx = n_0 sin(theta_0), branch Im q >= 0
  s:  r_ij = (q_i - q_j)/(q_i + q_j),            t_ij = 2 q_i/(q_i + q_j)
  p:  r_ij = (n_j^2 q_i - n_i^2 q_j)/(...+...),  t_ij = 2 n_i n_j q_i/(n_j^2 q_i + n_i^2 q_j)
  T = |t|^2 Re(q_out)/Re(q_in)  (s),   |t|^2 Re(n_out conj(q_out/n_out))/Re(n_in conj(q_in/n_in))  (p)
Thicknesses and wavelengths in nm.  Polarisation ``pol`` is a static Python int: 0 = s, 1 = p.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class SMat(NamedTuple):
    r: jax.Array    # reflection, incidence from the left (front/ambient side)
    t: jax.Array    # transmission left -> right
    rb: jax.Array   # reflection, incidence from the right (back side)
    tb: jax.Array   # transmission right -> left


class FacePowers(NamedTuple):
    R: jax.Array    # reflectance seen from the left medium
    T: jax.Array    # transmittance left -> right
    Rb: jax.Array   # reflectance seen from the right medium
    Tb: jax.Array   # transmittance right -> left


def kz(n, kx):
    """Normal wave-vector component (units of k0) on the decaying/forward branch Im q >= 0."""
    q = jnp.sqrt(n * n - kx * kx + 0j)
    flip = (jnp.imag(q) < 0) | ((jnp.imag(q) == 0) & (jnp.real(q) < 0))
    return jnp.where(flip, -q, q)


def interface(n_i, n_j, q_i, q_j, pol: int) -> SMat:
    if pol == 0:
        den = q_i + q_j
        r = (q_i - q_j) / den
        t = 2 * q_i / den
        tb = 2 * q_j / den
    else:
        ni2, nj2 = n_i * n_i, n_j * n_j
        den = nj2 * q_i + ni2 * q_j
        r = (nj2 * q_i - ni2 * q_j) / den
        t = 2 * n_i * n_j * q_i / den
        tb = 2 * n_i * n_j * q_j / den
    return SMat(r, t, -r, tb)


def star(A: SMat, B: SMat) -> SMat:
    """Redheffer star product: A on the left, B on the right."""
    inv = 1.0 / (1.0 - A.rb * B.r)
    return SMat(r=A.r + A.tb * B.r * A.t * inv,
                t=B.t * A.t * inv,
                rb=B.rb + B.t * A.rb * B.tb * inv,
                tb=A.tb * B.tb * inv)


def propagation(phase_factor) -> SMat:
    z = jnp.zeros_like(phase_factor)
    return SMat(z, phase_factor, z, phase_factor)


def flip(S: SMat) -> SMat:
    """Same physical stack seen from the other side."""
    return SMat(S.rb, S.tb, S.r, S.t)


def stack_smatrix(n_in, n_layers, n_out, d_nm, kx, lam_nm, pol: int):
    """S-matrix of ambient | layers | exit.  Returns (S, q_in, q_out). Zero layers allowed."""
    n_all = jnp.concatenate([jnp.atleast_1d(n_in), jnp.asarray(n_layers, dtype=complex), jnp.atleast_1d(n_out)])
    q_all = kz(n_all, kx)
    S0 = interface(n_all[0], n_all[1], q_all[0], q_all[1], pol)

    def body(S, xs):
        n_j, q_j, d_j, n_k, q_k = xs
        S = star(S, propagation(jnp.exp(2j * jnp.pi * q_j * d_j / lam_nm)))
        S = star(S, interface(n_j, n_k, q_j, q_k, pol))
        return S, None

    xs = (n_all[1:-1], q_all[1:-1], jnp.asarray(d_nm, dtype=float), n_all[2:], q_all[2:])
    S, _ = jax.lax.scan(body, S0, xs)
    return S, q_all[0], q_all[-1]


def abs2(z):
    return jnp.real(z) ** 2 + jnp.imag(z) ** 2   # smooth at z = 0 (jnp.abs is not)


def flux_factor(n, q, pol: int):
    return jnp.real(q) if pol == 0 else jnp.real(n * jnp.conj(q / n))


def _safe_ratio(a, b):
    ok = b > 1e-300
    return jnp.where(ok, a / jnp.where(ok, b, 1.0), 0.0)   # double-where: no NaN in gradients


def powers(S: SMat, n_in, q_in, n_out, q_out, pol: int) -> FacePowers:
    f_in, f_out = flux_factor(n_in, q_in, pol), flux_factor(n_out, q_out, pol)
    return FacePowers(R=abs2(S.r), T=abs2(S.t) * _safe_ratio(f_out, f_in),
                      Rb=abs2(S.rb), Tb=abs2(S.tb) * _safe_ratio(f_in, f_out))


def stack_powers(n_in, n_layers, n_out, d_nm, kx, lam_nm, pol: int) -> FacePowers:
    S, q_in, q_out = stack_smatrix(n_in, n_layers, n_out, d_nm, kx, lam_nm, pol)
    return powers(S, n_in, q_in, n_out, q_out, pol)


def sum_im_delta(n_layers, d_nm, kx, lam_nm):
    """Sum of Im(delta) over the layers: the exponent that drives T-matrix overflow."""
    return jnp.sum(jnp.imag(2 * jnp.pi * kz(jnp.asarray(n_layers, dtype=complex), kx) * d_nm / lam_nm))


class StableEngine:
    name = "stable (S-matrix, complex128)"

    def face(self, n_in, n_layers, n_out, d_nm, kx, lam_nm, pol: int) -> FacePowers:
        return stack_powers(n_in, n_layers, n_out, d_nm, kx, lam_nm, pol)
