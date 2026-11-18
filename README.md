# Interference filter optimization — two-wavelength 45° beam combiner

Two sources with fixed wavelengths illuminate opposite faces of a multilayer-coated plate at 45° (see `system.xml`).
The output ports are:
- **left** = {T1, R2}
- **up** = {R1, T2}

This project does two things:
1. **Step 1:** computes R, T and A for both sources.
2. **Step 2:** optimizes the coating thicknesses to maximize a chosen pair of coefficients (e.g. T1 + R2).

It is built on Python 3.14, JAX and tmmax. The full design rationale, the tmmax analysis and the measured findings are in
[docs/PLAN.md](docs/PLAN.md). The numerical-stability evidence is in `results/stability/`.

## Setup

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python3.14 -m venv .venv
```

```bash
.venv/bin/python -m pip install -e ".[test]"
```

tmmax is pinned to GitHub commit `dc4b904`. **PyPI tmmax 1.1.4 returns wrong R/T for stacks with two or more layers.**

## Usage

```bash
.venv/bin/python scripts/compute_rt.py --config config/example.toml
```

```bash
.venv/bin/python scripts/optimize_filter.py --config config/example.toml --maximize T1 R2
```

```bash
.venv/bin/python scripts/optimize_filter.py --config config/example.toml --port up --global-de
```

```bash
.venv/bin/python scripts/optimize_filter.py --config config/example_back_ar.toml --maximize T1 R2
```

```bash
.venv/bin/python scripts/compute_rt.py --config config/example.toml --design results/example_T1_R2/design.json
```

```bash
.venv/bin/python scripts/stability_study.py
```

```bash
.venv/bin/python -m pytest -q
```

Everything is set in `config/example.toml`:
- wavelengths and polarization per source (`s`, `p`, `unpolarized`, or an azimuth in degrees);
- substrate and its thickness, and which face carries the coating;
- an optional back-face coating: `[back_coating]` with `optimize = true` (optimized together with the front)
  or `optimize = false` (fixed thicknesses);
- the layer sequence and per-material thickness bounds;
- the objective (`maximize = ["T1","R2"]` or `port = "left"`; form `least_squares`, `log` or `softmin`; `both_polarizations`);
- optimizer settings.

Custom materials go under `[[materials.custom]]` as `constant`, `sellmeier` or `csv`. You can also drop a file
`materials/<Name>.csv` with the columns `wavelength_nm,n,k`.

## Layout

| Path | Purpose |
|---|---|
| `src/ifopt/engines/stable_engine.py` | S-matrix (Redheffer) solver in complex128. Stable for any loss or thickness, and differentiable. |
| `src/ifopt/engines/tmmax_engine.py` | tmmax adapter: in-memory nk data, runs under `enable_x64(False)`, makes the reversed-stack call. |
| `src/ifopt/engines/reference_engine.py` | Byrnes `tmm` (NumPy, float64), used as an independent reference. |
| `src/ifopt/plate.py` | Coated face + incoherent thick substrate + back face, polarization mixing, fringe envelope, ceilings. |
| `src/ifopt/objectives.py`, `optimize.py` | Merit functions, quarter-wave seed, vmapped multi-start Adam → L-BFGS-B, optional DE. |
| `src/ifopt/materials.py` | tmmax DB (read-only), Sellmeier glasses, CSV and constants. Raises outside the tabulated range. |
| `src/ifopt/geometry.py` | Port mapping, cross-checked against `system.xml`. |

## Key findings (details in docs/PLAN.md §12)

- **tmmax is single-precision only, and crashes when `jax_enable_x64` is on.**
- **With absorbing layers it fails early.** It returns NaN once Σ Im δ ≈ 89 (about 2.9 µm of Ge at 532 nm), and its gradients are wrong from Σ Im δ ≈ 46.
  - Even a complex128 T-matrix fails at Σ Im δ ≈ 710–765.
  - The S-matrix engine stays finite and accurate (≤ 1e-15 against Byrnes) for any thickness.
- **tmmax's incoherent-layer path applies the wrong layer's absorption.** For example it gives T = 0.44 where the true value is 0.0011. A closed-form formula is used instead.
- **Several tmmax database tables are unsuitable as they are:**
  - SiO2 has k = 0.002 at 532 nm.
  - Si only covers 2–20 µm and is silently clamped outside that range.
  - CdS has negative k.
- **Example design (bare back face):** T1 = 0.9174 and R2 = 0.9996 (s-pol). T1 is exactly at the ceiling set by the bare back face, 1 − R_s(45°).
- **With an optimized 4-layer back-face AR** (`config/example_back_ar.toml`): T1 = 0.999998 and R2 = 0.99973.
  - The back face also reflects 75% at 1064 nm, so R2 leaves as two beams about 0.8 mm apart.
  - T1 is more sensitive to 1% thickness errors (mean 0.983).
