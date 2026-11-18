"""Independent float64 reference: Byrnes' ``tmm`` package (NumPy transfer matrix with opacity cap)."""
from __future__ import annotations

import numpy as np

from .stable_engine import FacePowers


class ReferenceEngine:
    name = "reference (Byrnes tmm, float64)"

    def __init__(self):
        import tmm
        self._tmm = tmm

    def _rt(self, n_list, d_nm, kx, lam_nm, pol: int):
        n_list = list(np.asarray(n_list, dtype=complex))
        th0 = np.arcsin(kx / n_list[0])
        if abs(np.imag(th0)) < 1e-14:
            th0 = float(np.real(th0))
        res = self._tmm.coh_tmm("sp"[pol], n_list, [np.inf, *np.asarray(d_nm, float), np.inf], th0, lam_nm)
        return float(res["R"]), float(res["T"])

    def face(self, n_in, n_layers, n_out, d_nm, kx, lam_nm, pol: int) -> FacePowers:
        n_layers = np.asarray(n_layers, dtype=complex)
        d_nm = np.asarray(d_nm, dtype=float)
        fwd = np.concatenate([[n_in], n_layers, [n_out]])
        R, T = self._rt(fwd, d_nm, kx, lam_nm, pol)
        Rb, Tb = self._rt(fwd[::-1], d_nm[::-1], kx, lam_nm, pol)
        return FacePowers(R, T, Rb, Tb)
