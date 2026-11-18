"""Full plate model: front coating + thick (incoherent) substrate + back coating, both sources.

Plate (each face listed in ambient -> substrate order):
    ambient | front coating (face of source `coated_face`) | substrate, D | back coating | ambient

A design vector x holds the front thicknesses followed by the back thicknesses when the back face is
optimised (``Config.split``); a fixed back coating comes from the config.

For a source arriving on face a (exit face b):
    R_a, T_a   : face a seen from the ambient      R_a', T_a' : face a seen from the substrate
    R_b, T_b   : face b seen from the substrate
    tau        = exp(-4 pi D Im(q_s) / lambda)     single-pass internal transmittance (<= 1)

    R = R_a + T_a T_a' R_b tau^2 / (1 - R_a' R_b tau^2)
    T = T_a T_b tau / (1 - R_a' R_b tau^2)                 A = 1 - R - T

tau only multiplies, so an opaque substrate degrades smoothly to T -> 0, R -> R_a (tmmax's
incoherent path divides by tau).  Combination is done per polarisation; the mixed result uses
intensity weights (exact for isotropic stacks).

The coherent plate (substrate treated as a thin film) is evaluated over the substrate phase to
give the etalon fringe envelope; its phase average equals the incoherent result exactly.
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import Config
from .engines.stable_engine import abs2, flip, kz, powers, propagation, stack_smatrix, star

RTA = ("R", "T", "A")


class PlatePoint(NamedTuple):
    """One (source, wavelength, angle) evaluation point. All fields are arrays (vmap-able)."""
    lam: jax.Array          # nm
    kx: jax.Array           # n_ambient * sin(AOI)
    n_amb: jax.Array
    n_sub: jax.Array
    n_coat: jax.Array       # (L,) front coating
    n_back: jax.Array       # (M,) back coating
    D: jax.Array            # substrate thickness, nm
    coated_first: jax.Array # bool: source enters through the front (coated) face
    w: jax.Array            # (2,) polarisation weights (w_s, w_p)


def make_points(cfg: Config, source: int, lam_nm=None, aoi_deg=None) -> PlatePoint:
    """Batch of PlatePoints for one source over wavelengths and/or angles (broadcast together)."""
    lib = cfg.library
    lam = np.atleast_1d(np.asarray(cfg.sources[source].wavelength_nm if lam_nm is None else lam_nm, float))
    aoi = np.atleast_1d(np.asarray(cfg.aoi_deg if aoi_deg is None else aoi_deg, float))
    lam, aoi = np.broadcast_arrays(lam, aoi)
    lam, aoi = lam.ravel(), aoi.ravel()
    n_amb = lib.nk(cfg.ambient, lam)
    if np.any(np.abs(n_amb.imag) > 0):
        raise ValueError("ambient medium must be lossless")
    B = lam.size
    front, back = list(cfg.coating.materials), list(cfg.back.materials)
    return PlatePoint(
        lam=jnp.asarray(lam),
        kx=jnp.asarray(n_amb.real * np.sin(np.radians(aoi))),
        n_amb=jnp.asarray(n_amb),
        n_sub=jnp.asarray(lib.nk(cfg.plate.substrate, lam)),
        n_coat=jnp.asarray(lib.nk_table(front, lam).T.reshape(B, len(front))),
        n_back=jnp.asarray(lib.nk_table(back, lam).T.reshape(B, len(back))),
        D=jnp.full((B,), cfg.plate.substrate_thickness_mm * 1e6),
        coated_first=jnp.full((B,), cfg.plate.coated_face == source),
        w=jnp.asarray(np.broadcast_to(np.asarray(cfg.sources[source].pol_weights()), (B, 2))),
    )


def _faces(d_coat, d_back, pt: PlatePoint, pol: int):
    Sc, q_amb, q_sub = stack_smatrix(pt.n_amb, pt.n_coat, pt.n_sub, d_coat, pt.kx, pt.lam, pol)
    Sb, _, _ = stack_smatrix(pt.n_amb, pt.n_back, pt.n_sub, d_back, pt.kx, pt.lam, pol)
    return Sc, Sb, q_amb, q_sub


def combine_incoherent(Ra, Ta, Ra_in, Ta_in, Rb_in, Tb_in, tau):
    den = 1.0 - Ra_in * Rb_in * tau ** 2
    R = Ra + Ta * Ta_in * Rb_in * tau ** 2 / den
    T = Ta * Tb_in * tau / den
    return R, T, 1.0 - R - T


def internal_transmittance(pt: PlatePoint, q_sub):
    return jnp.exp(-4 * jnp.pi * pt.D * jnp.imag(q_sub) / pt.lam)


def _entrance_exit(pt, front, back):
    a = jax.tree.map(lambda c, b: jnp.where(pt.coated_first, c, b), front, back)
    b = jax.tree.map(lambda c, b_: jnp.where(pt.coated_first, b_, c), front, back)
    return a, b


def plate_point(d_coat, d_back, pt: PlatePoint) -> jax.Array:
    """(2 pols [s, p], 3 [R, T, A]) for one point, incoherent substrate. Differentiable in d_coat, d_back."""
    out = []
    for pol in (0, 1):
        Sc, Sb, q_amb, q_sub = _faces(d_coat, d_back, pt, pol)
        a, b = _entrance_exit(pt, powers(Sc, pt.n_amb, q_amb, pt.n_sub, q_sub, pol),
                              powers(Sb, pt.n_amb, q_amb, pt.n_sub, q_sub, pol))
        tau = internal_transmittance(pt, q_sub)
        out.append(jnp.stack(combine_incoherent(a.R, a.T, a.Rb, a.Tb, b.Rb, b.Tb, tau)))
    return jnp.stack(out)


def plate_point_coherent(d_coat, d_back, pt: PlatePoint, psi) -> jax.Array:
    """Coherent plate with the substrate single-pass phase replaced by psi (round trip 2 psi)."""
    out = []
    for pol in (0, 1):
        Sc, Sb, q_amb, q_sub = _faces(d_coat, d_back, pt, pol)
        Sa, Sx = _entrance_exit(pt, Sc, Sb)
        amp = jnp.sqrt(internal_transmittance(pt, q_sub)) * jnp.exp(1j * psi)
        S = star(star(Sa, propagation(amp)), flip(Sx))
        R, T = abs2(S.r), abs2(S.t)          # same ambient on both sides -> flux ratio 1
        out.append(jnp.stack([R, T, 1.0 - R - T]))
    return jnp.stack(out)


def mix(res_sp: jax.Array, w: jax.Array) -> jax.Array:
    """(..., 2, 3) per-polarisation results -> (..., 3) for the source's polarisation state."""
    return jnp.einsum("...p,...pk->...k", w, res_sp)


@jax.jit
def evaluate_points(d_coat, d_back, pts: PlatePoint) -> jax.Array:
    """Batch: (B, 2, 3)."""
    return jax.vmap(plate_point, in_axes=(None, None, 0))(d_coat, d_back, pts)


@partial(jax.jit, static_argnames="n_phase")
def fringe_envelope(d_coat, d_back, pts: PlatePoint, n_phase: int = 256):
    """min / mean / max over substrate phase of the coherent plate: each (B, 3) mixed."""
    psi = jnp.linspace(0.0, jnp.pi, n_phase, endpoint=False)
    f = jax.vmap(jax.vmap(plate_point_coherent, in_axes=(None, None, None, 0)), in_axes=(None, None, 0, None))
    res = mix(f(d_coat, d_back, pts, psi), pts.w[:, None, :])       # (B, n_phase, 3)
    return res.min(axis=1), res.mean(axis=1), res.max(axis=1)


class PlateModel:
    """The two operating points of the system (source 1 and source 2); methods take a design vector x."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        p1, p2 = make_points(cfg, 1), make_points(cfg, 2)
        self.points = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), p1, p2)  # batch of 2

    def per_pol(self, x) -> jax.Array:
        """(2 sources, 2 pols, 3 [R,T,A])."""
        return evaluate_points(*self.cfg.split(x), self.points)

    def mixed(self, x) -> jax.Array:
        """(2 sources, 3 [R,T,A]) with each source's polarisation state."""
        return mix(self.per_pol(x), self.points.w)

    def coefficients(self, x) -> dict[str, dict[str, float]]:
        res, m = np.asarray(self.per_pol(x)), np.asarray(self.mixed(x))
        out = {}
        for s in (1, 2):
            for k, q in enumerate(RTA):
                out[f"{q}{s}"] = {"s": float(res[s - 1, 0, k]), "p": float(res[s - 1, 1, k]), "mixed": float(m[s - 1, k])}
        return out

    def envelope(self, x, n_phase: int = 256):
        lo, mean, hi = fringe_envelope(*self.cfg.split(x), self.points, n_phase)
        return np.asarray(lo), np.asarray(mean), np.asarray(hi)

    def back_face_ceiling(self, x) -> np.ndarray:
        """Per source (mixed pol.): upper bound on the incoherent plate transmittance set by the back
        face as designed, T <= T_back * tau, valid for ANY passive front coating.
        (T = T_a T_b tau / (1 - R_a' R_b tau^2) with T_a <= 1 - R_a' is maximal at R_a' = 0.)"""
        _, d_back = self.cfg.split(x)

        def one(pt):
            vals = []
            for pol in (0, 1):
                Sb, q_amb, q_sub = stack_smatrix(pt.n_amb, pt.n_back, pt.n_sub, d_back, pt.kx, pt.lam, pol)
                P = powers(Sb, pt.n_amb, q_amb, pt.n_sub, q_sub, pol)
                vals.append(jnp.where(pt.coated_first, P.Tb, P.T) * internal_transmittance(pt, q_sub))
            return jnp.dot(pt.w, jnp.stack(vals))
        return np.asarray(jax.vmap(one)(self.points))

    def sum_im_delta(self, x) -> np.ndarray:
        """Per source: sum of Im(delta) over both coatings (T-matrix overflow exponent)."""
        d_coat, d_back = self.cfg.split(x)
        pts = self.points

        def s(n, d):
            q = kz(n, pts.kx[:, None])
            return jnp.sum(jnp.imag(2 * jnp.pi * q * jnp.asarray(d)[None, :] / pts.lam[:, None]), axis=1)
        return np.asarray(s(pts.n_coat, d_coat) + s(pts.n_back, d_back))


def evaluate_with_engine(cfg: Config, x, engine) -> dict[str, dict[str, float]]:
    """Same plate model, but the coherent faces are computed by an arbitrary engine
    (``TmmaxEngine`` / ``ReferenceEngine``); used for cross-validation."""
    out = {}
    d_coat, d_back = (np.asarray(a, float) for a in cfg.split(x))
    for s in (1, 2):
        pt = jax.tree.map(lambda a: np.asarray(a)[0], make_points(cfg, s))
        res = np.zeros((2, 3))
        for pol in (0, 1):
            Fc = engine.face(complex(pt.n_amb), pt.n_coat, complex(pt.n_sub), d_coat, float(pt.kx), float(pt.lam), pol)
            Fb = engine.face(complex(pt.n_amb), pt.n_back, complex(pt.n_sub), d_back, float(pt.kx), float(pt.lam), pol)
            a, b = (Fc, Fb) if bool(pt.coated_first) else (Fb, Fc)
            q_sub = complex(kz(jnp.asarray(pt.n_sub), pt.kx))
            tau = float(np.exp(-4 * np.pi * float(pt.D) * q_sub.imag / float(pt.lam)))
            res[pol] = [float(v) for v in combine_incoherent(float(a.R), float(a.T), float(a.Rb), float(a.Tb),
                                                             float(b.Rb), float(b.Tb), tau)]
        m = np.asarray(pt.w) @ res
        for k, q in enumerate(RTA):
            out[f"{q}{s}"] = {"s": res[0, k], "p": res[1, k], "mixed": m[k]}
    return out
