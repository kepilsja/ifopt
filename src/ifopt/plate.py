"""Evaluation model: front coating, optional substrate and back coating, any set of target points.

Substrate modes (``[substrate] mode``):

  incoherent    ambient | front | substrate D | back | ambient   - thick substrate, intensities added
  coherent      same, substrate as an etalon at its physical phase (single-frequency analysis)
  semi_infinite ambient | front | substrate                      - back surface ignored ("substrate off")
  none          ambient | front | exit_medium                    - free-standing stack

``side = front``: light arrives from the ambient on the front coating.  ``side = back``: on the
back face from the ambient (plate modes) or from the substrate / exit medium (other modes).
kx = n_incidence * sin(AOI) is taken in the incidence medium.

Incoherent plate, light entering face a and leaving through face b:
    R = R_a + T_a T_a' R_b tau^2 / (1 - R_a' R_b tau^2),   T = T_a T_b tau / (1 - R_a' R_b tau^2)
with tau = exp(-4 pi D Im(q_s) / lambda) <= 1 (never divided by, so opaque substrates are safe).
The coherent plate's phase average equals the incoherent result exactly (used for the fringe envelope).
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import PLATE_MODES, Config
from .engines.stable_engine import abs2, flip, kz, powers, propagation, stack_smatrix, star
from .targets import TargetTable

RTA = ("R", "T", "A")


class EvalPoint(NamedTuple):
    """One (wavelength, AOI, side) evaluation point; all fields are arrays (vmap-able)."""
    lam: jax.Array          # nm
    kx: jax.Array           # n_incidence * sin(AOI)
    n_amb: jax.Array
    n_sub: jax.Array        # substrate, or exit medium in mode "none"
    n_coat: jax.Array       # (L,) front coating
    n_back: jax.Array       # (M,) back coating
    D: jax.Array            # substrate thickness, nm (plate modes)
    front_side: jax.Array   # bool: light arrives on the front side


def make_points(cfg: Config, lam_nm, aoi_deg, front_side) -> EvalPoint:
    lib = cfg.library
    lam, aoi, front = (np.ravel(a) for a in np.broadcast_arrays(np.asarray(lam_nm, float), np.asarray(aoi_deg, float),
                                                                 np.asarray(front_side, bool)))
    B = lam.size
    n_amb = lib.nk(cfg.ambient, lam)
    n_sub = lib.nk(cfg.exit_material, lam)
    if np.any(n_amb.imag != 0):
        raise ValueError("ambient medium must be lossless")
    n_inc = n_amb if cfg.has_back_face else np.where(front, n_amb, n_sub)
    if np.any(n_inc.imag != 0):
        raise ValueError(f"back-side incidence from an absorbing medium ({cfg.exit_material}) is not supported")
    fm, bm = list(cfg.front.materials), list(cfg.back.materials)
    return EvalPoint(
        lam=jnp.asarray(lam), kx=jnp.asarray(n_inc.real * np.sin(np.radians(aoi))),
        n_amb=jnp.asarray(n_amb), n_sub=jnp.asarray(n_sub),
        n_coat=jnp.asarray(lib.nk_table(fm, lam).T.reshape(B, len(fm))),
        n_back=jnp.asarray(lib.nk_table(bm, lam).T.reshape(B, len(bm))),
        D=jnp.full((B,), cfg.substrate.thickness_mm * 1e6 if cfg.has_back_face else 0.0),
        front_side=jnp.asarray(front))


# ------------------------------------------------------------------------------ physics --
def _faces(d_coat, d_back, pt: EvalPoint, pol: int):
    Sc, q_amb, q_sub = stack_smatrix(pt.n_amb, pt.n_coat, pt.n_sub, d_coat, pt.kx, pt.lam, pol)
    Sb, _, _ = stack_smatrix(pt.n_amb, pt.n_back, pt.n_sub, d_back, pt.kx, pt.lam, pol)
    return Sc, Sb, q_amb, q_sub


def combine_incoherent(Ra, Ta, Ra_in, Ta_in, Rb_in, Tb_in, tau):
    den = 1.0 - Ra_in * Rb_in * tau ** 2
    R = Ra + Ta * Ta_in * Rb_in * tau ** 2 / den
    T = Ta * Tb_in * tau / den
    return R, T, 1.0 - R - T


def internal_transmittance(pt: EvalPoint, q_sub):
    return jnp.exp(-4 * jnp.pi * pt.D * jnp.imag(q_sub) / pt.lam)


def _entrance_exit(pt, front, back):
    a = jax.tree.map(lambda c, b: jnp.where(pt.front_side, c, b), front, back)
    b = jax.tree.map(lambda c, b_: jnp.where(pt.front_side, b_, c), front, back)
    return a, b


def plate_point(d_coat, d_back, pt: EvalPoint) -> jax.Array:
    """Incoherent plate: (2 pols [s, p], 3 [R, T, A])."""
    out = []
    for pol in (0, 1):
        Sc, Sb, q_amb, q_sub = _faces(d_coat, d_back, pt, pol)
        a, b = _entrance_exit(pt, powers(Sc, pt.n_amb, q_amb, pt.n_sub, q_sub, pol),
                              powers(Sb, pt.n_amb, q_amb, pt.n_sub, q_sub, pol))
        out.append(jnp.stack(combine_incoherent(a.R, a.T, a.Rb, a.Tb, b.Rb, b.Tb, internal_transmittance(pt, q_sub))))
    return jnp.stack(out)


def plate_point_coherent(d_coat, d_back, pt: EvalPoint, psi) -> jax.Array:
    """Coherent plate with the substrate single-pass phase set to psi (round trip 2 psi)."""
    out = []
    for pol in (0, 1):
        Sc, Sb, q_amb, q_sub = _faces(d_coat, d_back, pt, pol)
        Sa, Sx = _entrance_exit(pt, Sc, Sb)
        amp = jnp.sqrt(internal_transmittance(pt, q_sub)) * jnp.exp(1j * psi)
        S = star(star(Sa, propagation(amp)), flip(Sx))
        R, T = abs2(S.r), abs2(S.t)                  # same ambient on both sides -> flux ratio 1
        out.append(jnp.stack([R, T, 1.0 - R - T]))
    return jnp.stack(out)


def physical_phase(pt: EvalPoint):
    return 2 * jnp.pi * pt.D * jnp.real(kz(pt.n_sub, pt.kx)) / pt.lam


def coherent_point(d_coat, d_back, pt: EvalPoint) -> jax.Array:
    return plate_point_coherent(d_coat, d_back, pt, physical_phase(pt))


def stack_point(d_coat, d_back, pt: EvalPoint) -> jax.Array:
    """Front coating between ambient and a semi-infinite substrate / exit medium."""
    out = []
    for pol in (0, 1):
        S, q_in, q_out = stack_smatrix(pt.n_amb, pt.n_coat, pt.n_sub, d_coat, pt.kx, pt.lam, pol)
        P = powers(S, pt.n_amb, q_in, pt.n_sub, q_out, pol)
        R = jnp.where(pt.front_side, P.R, P.Rb)
        T = jnp.where(pt.front_side, P.T, P.Tb)
        out.append(jnp.stack([R, T, 1.0 - R - T]))
    return jnp.stack(out)


RESPONSES = {"incoherent": plate_point, "coherent": coherent_point, "semi_infinite": stack_point, "none": stack_point}


@partial(jax.jit, static_argnames="mode")
def evaluate(d_coat, d_back, pts: EvalPoint, mode: str) -> jax.Array:
    """(P, 2 pols, 3 [R, T, A])."""
    return jax.vmap(RESPONSES[mode], in_axes=(None, None, 0))(d_coat, d_back, pts)


@partial(jax.jit, static_argnames="n_phase")
def coherent_scan(d_coat, d_back, pts: EvalPoint, n_phase: int = 256):
    """Coherent plate over the substrate single-pass phase psi in [0, pi): (P, n_phase, 2, 3).
    Mix polarisations first, then take min / max (the fringe envelope); the mean equals the
    incoherent result."""
    psi = jnp.linspace(0.0, jnp.pi, n_phase, endpoint=False)
    f = jax.vmap(jax.vmap(plate_point_coherent, in_axes=(None, None, None, 0)), in_axes=(None, None, 0, None))
    return f(d_coat, d_back, pts, psi)


def mix(res_sp, w):
    """(..., 2, 3) per polarisation -> (..., 3) with weights (..., 2)."""
    return jnp.einsum("...p,...pk->...k", w, res_sp)


# -------------------------------------------------------------------------------- model --
class Model:
    """All evaluation points of the target table; methods take a design vector x."""

    def __init__(self, cfg: Config, table: TargetTable | None = None):
        self.cfg = cfg
        self.table = table or cfg.table
        self.mode = cfg.mode
        t = self.table
        self.points = make_points(cfg, t.point_lam, t.point_aoi, t.point_front)
        self.t_point, self.t_q = jnp.asarray(t.t_point), jnp.asarray(t.t_q)
        self.t_polw = jnp.asarray(t.t_polw)

    def response(self, x) -> jax.Array:
        return evaluate(*self.cfg.split(x), self.points, self.mode)

    def target_values(self, x, res=None) -> jax.Array:
        res = self.response(x) if res is None else res
        return jnp.einsum("kp,kp->k", self.t_polw, res[self.t_point, :, self.t_q])

    def beam_response(self, x, label: str, res=None) -> tuple[np.ndarray, np.ndarray]:
        """(n, 2, 3) per polarisation and (n, 3) mixed with the beam's polarisation, over its points."""
        res = np.asarray(self.response(x) if res is None else res)
        b = self.table.beams[label]
        per = res[b.points]
        return per, np.einsum("p,npk->nk", np.asarray(b.weights), per)

    def coherent_scan(self, x, n_phase: int = 256):
        """(P, n_phase, 2, 3) or None outside the plate modes."""
        if not self.cfg.has_back_face:
            return None
        return np.asarray(coherent_scan(*self.cfg.split(x), self.points, n_phase))

    def back_face_ceiling(self, x):
        """(P, 2): incoherent-plate transmittance bound set by the back face as designed,
        T <= T_back * tau, valid for any passive front coating (plate modes only)."""
        if self.cfg.mode != "incoherent":
            return None
        _, d_back = self.cfg.split(x)

        def one(pt):
            vals = []
            for pol in (0, 1):
                Sb, q_amb, q_sub = stack_smatrix(pt.n_amb, pt.n_back, pt.n_sub, d_back, pt.kx, pt.lam, pol)
                P = powers(Sb, pt.n_amb, q_amb, pt.n_sub, q_sub, pol)
                vals.append(jnp.where(pt.front_side, P.Tb, P.T) * internal_transmittance(pt, q_sub))
            return jnp.stack(vals)
        return np.asarray(jax.vmap(one)(self.points))

    def sum_im_delta(self, x) -> np.ndarray:
        """(P,): sum of Im(delta) over both coatings (T-matrix overflow exponent)."""
        d_coat, d_back = self.cfg.split(x)
        pts = self.points

        def s(n, d):
            q = kz(n, pts.kx[:, None])
            return jnp.sum(jnp.imag(2 * jnp.pi * q * jnp.asarray(d)[None, :] / pts.lam[:, None]), axis=1)
        return np.asarray(s(pts.n_coat, d_coat) + s(pts.n_back, d_back))


def evaluate_with_engine(cfg: Config, x, engine, point_idx=None) -> np.ndarray:
    """(len(point_idx), 2, 3) with the coherent faces from another engine (tmmax / reference).
    Same mode-specific combination as the stable model; the coherent mode is not supported."""
    if cfg.mode == "coherent":
        raise NotImplementedError("engine cross-check is not available in coherent mode")
    t = cfg.table
    idx = np.arange(t.n_points) if point_idx is None else np.asarray(point_idx)
    d_coat, d_back = (np.asarray(a, float) for a in cfg.split(x))
    pts = make_points(cfg, t.point_lam[idx], t.point_aoi[idx], t.point_front[idx])
    out = np.zeros((idx.size, 2, 3))
    for i in range(idx.size):
        pt = jax.tree.map(lambda a: np.asarray(a)[i], pts)
        args = (complex(pt.n_amb), pt.n_coat, complex(pt.n_sub), d_coat, float(pt.kx), float(pt.lam))
        for pol in (0, 1):
            Fc = engine.face(*args, pol)
            if cfg.mode in PLATE_MODES:
                Fb = engine.face(complex(pt.n_amb), pt.n_back, complex(pt.n_sub), d_back, float(pt.kx), float(pt.lam), pol)
                a, b = (Fc, Fb) if bool(pt.front_side) else (Fb, Fc)
                q_sub = complex(kz(jnp.asarray(pt.n_sub), pt.kx))
                tau = float(np.exp(-4 * np.pi * float(pt.D) * q_sub.imag / float(pt.lam)))
                out[i, pol] = [float(v) for v in combine_incoherent(float(a.R), float(a.T), float(a.Rb), float(a.Tb),
                                                                    float(b.Rb), float(b.Tb), tau)]
            else:
                R, T = (Fc.R, Fc.T) if bool(pt.front_side) else (Fc.Rb, Fc.Tb)
                out[i, pol] = [float(R), float(T), 1.0 - float(R) - float(T)]
    return out
