"""Numerical-stability study of tmmax vs the S-matrix engine for highly lossy layers.

    python scripts/stability_study.py [--config config/combiner_45.toml | --wavelengths 532 1064] [--out results/stability]

Sections (written to <out>/stability_report.md and <out>/stability.png):
  1. single absorbing layer, thickness sweep: R, T and dR/dd from
       tmmax (T-matrix, complex64) | own T-matrix in complex128 without capping |
       Byrnes tmm (T-matrix, float64, Im(delta) capped at 35) | stable S-matrix (complex128)
  2. tmmax incoherent-layer path vs Byrnes inc_tmm vs our closed-form thick-layer formula
  3. tmmax compatibility checks (x64 mode, jit(grad(tmm())), bare interface)
  4. tmmax nk-database audit at the configured wavelengths
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import ifopt  # noqa: E402,F401  (enables x64)
from ifopt.config import load_config  # noqa: E402
from ifopt.engines.stable_engine import kz, stack_powers  # noqa: E402
from ifopt.materials import MaterialLibrary, tmmax_db_dir  # noqa: E402
from ifopt.plate import combine_incoherent  # noqa: E402

AOI = np.pi / 4


# ---------------------------------------------------------------- engines for one layer --
def tmatrix_c128(n_list, d_nm, kx, lam, pol):
    """Plain Byrnes-style T-matrix in complex128, NO opacity cap (to show the float64 wall)."""
    n = np.asarray(n_list, complex)
    q = np.sqrt(n * n - kx * kx + 0j)
    q = np.where(q.imag < 0, -q, q)
    def rt(i, j):
        if pol == 0:
            return (q[i] - q[j]) / (q[i] + q[j]), 2 * q[i] / (q[i] + q[j])
        den = n[j] ** 2 * q[i] + n[i] ** 2 * q[j]
        return (n[j] ** 2 * q[i] - n[i] ** 2 * q[j]) / den, 2 * n[i] * n[j] * q[i] / den
    with np.errstate(all="ignore"):
        r0, t0 = rt(0, 1)
        M = np.array([[1, r0], [r0, 1]]) / t0
        for j in range(1, len(n) - 1):
            delta = 2 * np.pi * q[j] * d_nm[j - 1] / lam
            r, t = rt(j, j + 1)
            M = M @ (np.array([[np.exp(-1j * delta), 0], [0, np.exp(1j * delta)]]) @ np.array([[1, r], [r, 1]]) / t)
        rr, tt = M[1, 0] / M[0, 0], 1 / M[0, 0]
        f = (lambda m: q[m].real) if pol == 0 else (lambda m: (n[m] * np.conj(q[m] / n[m])).real)
        return abs(rr) ** 2, abs(tt) ** 2 * f(-1) / f(0)


def tmmax_layer(n_list, d_nm, lam, pol, grad=False):
    from tmmax.tmm import vectorized_coh_tmm
    n_list = np.asarray(n_list, complex)
    with jax.enable_x64(False):
        lam_m = lam * 1e-9
        data = np.empty((n_list.size, 3, 2), np.float32)
        data[:, 0, :] = [lam_m * 0.999, lam_m * 1.001]
        data[:, 1, :] = n_list.real[:, None]
        data[:, 2, :] = n_list.imag[:, None]
        args = (jnp.asarray(data), jnp.arange(n_list.size))
        rest = (jnp.asarray([lam_m], jnp.float32), jnp.asarray([AOI], jnp.float32), jnp.array([bool(pol)]))
        d = jnp.asarray(np.asarray(d_nm) * 1e-9, jnp.float32)
        R, T = vectorized_coh_tmm(*args, d, *rest)
        out = [float(np.ravel(R)[0]), float(np.ravel(T)[0])]
        if grad:
            g = jax.grad(lambda x: vectorized_coh_tmm(*args, x, *rest)[0].sum())(d)
            out.append(float(np.ravel(g)[0]) * 1e-9)   # per nm
        return out


def sweep(name, n_layer, lam, n_sub, thicknesses, pol=0):
    import tmm as bt
    kx = np.sin(AOI)
    n_list = [1.0 + 0j, n_layer, n_sub]
    q = complex(kz(jnp.asarray(n_layer), kx))
    gs = jax.jit(jax.grad(lambda d: stack_powers(1.0 + 0j, jnp.asarray([n_layer]), n_sub, d, kx, lam, pol).R[()].sum()))
    rows = []
    for d in thicknesses:
        sp = stack_powers(1.0 + 0j, jnp.asarray([n_layer]), n_sub, jnp.asarray([d]), kx, lam, pol)
        g_s = float(gs(jnp.asarray([d]))[0])
        Rm, Tm, gm = tmmax_layer(n_list, [d], lam, pol, grad=True)
        R64, T64 = tmatrix_c128(n_list, [d], kx, lam, pol)
        ref = bt.coh_tmm("sp"[pol], n_list, [np.inf, d, np.inf], AOI, lam)
        rows.append(dict(d=d, im_delta=2 * np.pi * q.imag * d / lam,
                         R_stable=float(sp.R), T_stable=float(sp.T), g_stable=g_s,
                         R_tmmax=Rm, T_tmmax=Tm, g_tmmax=gm, R_c128=R64, T_c128=T64,
                         R_byrnes=float(ref["R"]), T_byrnes=float(ref["T"])))
    return dict(name=name, lam=lam, n=n_layer, q=q, rows=rows)


def first_failure(rows, key, ref_key, tol):
    for r in rows:
        v = r[key]
        if not np.isfinite(v) or abs(v - r[ref_key]) > tol:
            return r
    return None


# ------------------------------------------------------------------- incoherent path ------
def incoherent_cases(lib):
    import tmm as bt
    from tmmax.tmm import tmm as tmmax_tmm
    cases = [(["Air", "Ge", "TiO2", "Air"], [2000.0, 100.0], [1, 0], 1064.0),
             (["Air", "Ge", "TiO2", "Ge", "TiO2", "Air"], [2000.0, 100.0, 2000.0, 100.0], [1, 0, 1, 0], 1064.0),
             (["Air", "Ge", "TiO2", "Air"], [20000.0, 100.0], [1, 0], 532.0)]
    out = []
    for mats, th, coh, lam in cases:
        nk = [complex(lib.nk(m, lam)[0]) for m in mats]
        with jax.enable_x64(False):
            try:
                R, T = tmmax_tmm(mats, jnp.asarray(np.array(th) * 1e-9, jnp.float32), jnp.asarray([lam * 1e-9], jnp.float32),
                                 jnp.asarray([AOI], jnp.float32), coherency_list=jnp.asarray(coh), polarization="s")
                got = (float(np.ravel(R)[0]), float(np.ravel(T)[0]))
            except Exception as exc:  # pragma: no cover
                got = (f"error {type(exc).__name__}", "")
        ref = bt.inc_tmm("s", nk, [np.inf, *th, np.inf], ["i", *["i" if c else "c" for c in coh], "i"], AOI, lam)
        ours = None
        if len(mats) == 4:   # ambient | thick (incoherent) | thin coherent | ambient  -> closed form
            kx = np.sin(AOI)
            fa = stack_powers(nk[0], jnp.zeros(0, complex), nk[1], jnp.zeros(0), kx, lam, 0)
            fb = stack_powers(nk[3], jnp.asarray([nk[2]]), nk[1], jnp.asarray([th[1]]), kx, lam, 0)
            tau = float(np.exp(-4 * np.pi * th[0] * complex(kz(jnp.asarray(nk[1]), kx)).imag / lam))
            R_, T_, _ = combine_incoherent(float(fa.R), float(fa.T), float(fa.Rb), float(fa.Tb), float(fb.Rb), float(fb.Tb), tau)
            ours = (float(R_), float(T_))
        out.append(dict(mats=mats, th=th, coh=coh, lam=lam, tmmax=got, ref=(float(ref["R"]), float(ref["T"])), ours=ours))
    return out


# -------------------------------------------------------------- compatibility checks ------
def compat_checks():
    from tmmax.tmm import tmm as tmmax_tmm
    res = {}
    mats, d = ["Air", "TiO2", "SiO2", "TiO2", "Air"], jnp.asarray([60e-9, 110e-9, 70e-9])
    args = (jnp.asarray([532e-9]), jnp.asarray([AOI]))
    try:
        tmmax_tmm(mats, d, *args, polarization="s")
        res["x64 enabled"] = "works"
    except Exception as exc:
        res["x64 enabled"] = f"FAILS: {type(exc).__name__}: {str(exc).splitlines()[0][:110]}"
    with jax.enable_x64(False):
        d32 = jnp.asarray([60e-9, 110e-9, 70e-9], jnp.float32)
        a32 = (jnp.asarray([532e-9], jnp.float32), jnp.asarray([AOI], jnp.float32))
        try:
            jax.grad(lambda x: tmmax_tmm(mats, x, *a32, polarization="s")[0].sum())(d32)
            res["grad(tmm()) eager"] = "works"
        except Exception as exc:
            res["grad(tmm()) eager"] = f"FAILS: {type(exc).__name__}"
        try:
            jax.jit(jax.grad(lambda x: tmmax_tmm(mats, x, *a32, polarization="s")[0].sum()))(d32)
            res["jit(grad(tmm()))"] = "works"
        except Exception as exc:
            res["jit(grad(tmm()))"] = f"FAILS: {type(exc).__name__} (np.any input validation on traced values)"
        try:
            tmmax_tmm(["Air", "SiO2"], jnp.zeros(0, jnp.float32), *a32, polarization="s")
            res["bare interface (0 layers)"] = "works"
        except Exception as exc:
            res["bare interface (0 layers)"] = f"FAILS: {type(exc).__name__}"
        R, _ = tmmax_tmm(mats, d32, *a32, polarization="s")
        res["output dtype"] = str(R.dtype)
    return res


def db_audit(wavelengths):
    rows = []
    for p in sorted(tmmax_db_dir().glob("*.npy")):
        wl_um, n, k = np.load(p)
        rng = (wl_um.min() * 1e3, wl_um.max() * 1e3)
        vals, neg_k = [], bool(np.any(k < 0))
        for lam in wavelengths:
            if not (rng[0] <= lam <= rng[1]):
                vals.append(None)
            else:
                vals.append(complex(np.interp(lam, wl_um * 1e3, n) + 1j * np.interp(lam, wl_um * 1e3, k)))
        rows.append((p.stem, rng, vals, neg_k))
    return rows


# ---------------------------------------------------------------------------- report -----
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config/combiner_45.toml"),
                    help="database audit at the centre wavelengths of this config's beams")
    ap.add_argument("--wavelengths", nargs="+", type=float, help="database audit wavelengths (nm), overrides --config")
    ap.add_argument("--out", default=str(ROOT / "results/stability"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    lib = MaterialLibrary()
    n_fs = complex(lib.nk("FusedSilica", 532.0)[0])

    thick = np.unique(np.concatenate([np.geomspace(10, 60000, 60), [1000, 1500, 2000, 2500, 2900, 3000]]))
    studies = [sweep("Ge (tmmax DB) @ 532 nm", complex(lib.nk("Ge", 532.0)[0]), 532.0, n_fs, thick),
               sweep("metal-like n = 0.05 + 3.4i @ 532 nm (illustrative)", 0.05 + 3.4j, 532.0, n_fs, thick)]

    L = ["# Numerical stability study (generated by scripts/stability_study.py)", "",
         "Single absorbing layer between air and fused silica, 45 deg, s-polarisation.",
         "Engines: **tmmax** (T-matrix, complex64), **T-matrix c128** (same algorithm, complex128, no cap), "
         "**Byrnes tmm** (T-matrix float64, caps Im(delta) at 35), **stable** (S-matrix, complex128).", ""]
    for st in studies:
        rows = st["rows"]
        L.append(f"## {st['name']}  (q = {st['q']:.3f}, Im(delta) per nm = {2 * np.pi * st['q'].imag / st['lam']:.4f})")
        f_t = first_failure(rows, "R_tmmax", "R_stable", 1e-4)
        f_g = first_failure(rows, "g_tmmax", "g_stable", 1e-4)
        f_64 = first_failure(rows, "R_c128", "R_stable", 1e-9)
        L.append(f"- tmmax R/T first wrong or non-finite at d = {f_t['d']:.0f} nm (sum Im delta = {f_t['im_delta']:.1f}); "
                 f"value R = {f_t['R_tmmax']}" if f_t else "- tmmax R/T correct over the whole sweep")
        L.append(f"- tmmax gradient dR/dd first wrong at d = {f_g['d']:.0f} nm (sum Im delta = {f_g['im_delta']:.1f}); "
                 f"dR/dd = {f_g['g_tmmax']:.3g} /nm vs {f_g['g_stable']:.3g} /nm" if f_g else "- tmmax gradient correct")
        L.append(f"- complex128 T-matrix (no cap) first fails at d = {f_64['d']:.0f} nm (sum Im delta = {f_64['im_delta']:.1f})"
                 if f_64 else "- complex128 T-matrix fine over the sweep")
        L.append(f"- stable S-matrix: all values finite = {all(np.isfinite(r['R_stable']) and np.isfinite(r['g_stable']) for r in rows)}; "
                 f"max |R - R_Byrnes| = {max(abs(r['R_stable'] - r['R_byrnes']) for r in rows):.1e}")
        L.append("")
        L.append("| d (nm) | sum Im δ | R stable | R tmmax | R c128 | T stable | T tmmax | dR/dd stable | dR/dd tmmax |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            if r["d"] in (10, 100, 1000, 1500, 2000, 2500, 2900, 3000) or r is rows[-1] or abs(np.log10(r["d"]) % 1) < 0.03:
                L.append(f"| {r['d']:.0f} | {r['im_delta']:.1f} | {r['R_stable']:.6f} | {r['R_tmmax']:.6g} | {r['R_c128']:.6g} | "
                         f"{r['T_stable']:.3g} | {r['T_tmmax']:.3g} | {r['g_stable']:.3g} | {r['g_tmmax']:.3g} |")
        L.append("")

    L.append("## tmmax incoherent-layer path (s-pol, 45 deg)")
    L.append("| stack | coherency | λ (nm) | tmmax R, T | Byrnes inc_tmm R, T | closed form (ours) R, T |")
    L.append("|---|---|---:|---|---|---|")
    for c in incoherent_cases(lib):
        fmt = lambda p: "-" if p is None else (f"{p[0]:.4g}, {p[1]:.3g}" if isinstance(p[0], float) else str(p[0]))
        L.append(f"| {' / '.join(f'{m} {t:g}' for m, t in zip(c['mats'][1:-1], c['th']))} | {c['coh']} | {c['lam']:g} | "
                 f"{fmt(c['tmmax'])} | {fmt(c['ref'])} | {fmt(c['ours'])} |")
    L.append("")
    L.append("## tmmax compatibility checks")
    for k, v in compat_checks().items():
        L.append(f"- {k}: {v}")
    L.append("")
    lams = args.wavelengths or sorted({round(0.5 * (min(b.wavelengths_nm) + max(b.wavelengths_nm)), 3)
                                       for b in cfg.table.beams.values()})
    L.append(f"## tmmax nk database at {' / '.join(f'{v:g}' for v in lams)} nm")
    L.append("Out of range = tmmax would silently clamp to the table end. k > 1e-3 flagged for dielectrics.")
    L.append("")
    L.append("| material | table range (nm) | " + " | ".join(f"N at {v:g} nm" for v in lams) + " | flag |")
    L.append("|---|---|" + "---|" * len(lams) + "---|")
    for name, rng, vals, neg_k in db_audit(lams):
        flag = ["**negative k in table (gain)**"] if neg_k else []
        if any(v is None for v in vals):
            flag.append("**out of range (silent clamp)**")
        if any(v is not None and 1e-3 < v.imag < 0.1 for v in vals):
            flag.append("k > 1e-3")
        f = lambda v: "out of range" if v is None else f"{v.real:.4f}+{v.imag:.2e}i"
        L.append(f"| {name} | {rng[0]:.0f}-{rng[1]:.0f} | " + " | ".join(f(v) for v in vals) + f" | {', '.join(flag)} |")
    (out / "stability_report.md").write_text("\n".join(L) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, st in enumerate(studies):
        rows = st["rows"]
        d = np.array([r["d"] for r in rows]) / 1000
        ax = axes[0, col]
        for key, lab, sty in [("R_stable", "S-matrix c128", "-"), ("R_tmmax", "tmmax c64", "o"), ("R_c128", "T-matrix c128", "x"),
                              ("R_byrnes", "Byrnes (capped)", ":")]:
            v = np.array([r[key] for r in rows], float)
            ax.semilogx(d, np.where(np.isfinite(v), v, np.nan), sty, ms=3, label=lab)
            bad = ~np.isfinite(v)
            if bad.any():
                ax.semilogx(d[bad], np.full(bad.sum(), 0.02 + 0.03 * col), "|", ms=8, label=f"{lab}: NaN/inf")
        ax.set(xlabel="layer thickness (µm)", ylabel="R", title=st["name"], ylim=(0, 1))
        ax.legend(fontsize=7)
        ax = axes[1, col]
        for key, lab, sty in [("g_stable", "S-matrix c128", "-"), ("g_tmmax", "tmmax c64", "o")]:
            v = np.array([r[key] for r in rows], float)
            ax.semilogx(d, np.abs(np.where(np.isfinite(v), v, np.nan)) + 1e-30, sty, ms=3, label=lab)
        ax.set_yscale("log")
        ax.set(xlabel="layer thickness (µm)", ylabel="|dR/dd| (1/nm)", title="gradient (NaN not drawn)")
        ax.legend(fontsize=7)
    fig.savefig(out / "stability.png", dpi=130)
    print((out / "stability_report.md").read_text())
    print(f"Saved {out / 'stability_report.md'} and {out / 'stability.png'}")


if __name__ == "__main__":
    main()
