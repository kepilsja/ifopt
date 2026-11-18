# Interference filter optimization

This project designs multilayer interference filters with thin-film design-software conventions:
- **Specification:** a **target table** of required R, T or A values at given wavelengths and angles of incidence,
  for given polarizations and sides of the filter.
- **Evaluation:** a numerically stable S-matrix engine (JAX, complex128).
- **Optimization:** layer thicknesses, using a multi-start gradient search followed by an L-BFGS-B polish.
- **Cross-checks:** tmmax and Byrnes' `tmm`.

Included examples:
- the 45° two-source **beam combiner** from `system.xml` (sources on opposite faces; ports left = {T@1, R@2} and
  up = {R@1, T@2});
- a **normal-incidence dichroic** with wavelength bands, on a semi-infinite substrate (substrate "off").

The design rationale and measured numbers are in [docs/DESIGN.md](docs/DESIGN.md). The numerical-stability evidence is
in [results/stability/stability_report.md](results/stability/stability_report.md).

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
.venv/bin/python scripts/compute_rt.py --config config/combiner_45.toml
```

```bash
.venv/bin/python scripts/optimize_filter.py --config config/combiner_45.toml --out results/combiner_45
```

```bash
.venv/bin/python scripts/optimize_filter.py --config config/normal_dichroic_bands.toml --out results/normal_dichroic_bands
```

```bash
.venv/bin/python scripts/compute_rt.py --config config/combiner_45.toml --design results/combiner_45/design.json
```

```bash
.venv/bin/python scripts/stability_study.py
```

```bash
.venv/bin/python -m pytest -q
```

`compute_rt.py` evaluates the configured thicknesses, a saved `design.json`, or the quarter-wave seed. Its report covers:
- R, T and A per beam (s, p, and the beam's polarization), with the differences against tmmax and Byrnes;
- in the plate modes, the coherent fringe envelope and the back-face ceiling;
- the target table with each target's achieved value;
- the merit and figure of merit;
- the configured sums.

`optimize_filter.py` writes `design.json`, `report.txt` (including a thickness Monte Carlo) and per-beam spectra plots.
Pass `--starts`, `--steps` or `--global-de` to override the optimizer settings.

## Config schema

```toml
[system]
ambient = "Air"

[substrate]
material = "FusedSilica"
mode = "incoherent"            # incoherent | coherent | semi_infinite | none
thickness_mm = 1.0             # incoherent / coherent
exit_medium = "Air"            # mode = "none" (default: ambient)

[front]                        # coating on the front face, ambient -> substrate order
materials = ["TiO2", "FusedSilica", "TiO2"]
thickness_nm = [100, 180, 100] # optional: design to evaluate / extra starting point
optimize = true
[front.bounds_nm]
default = [5.0, 400.0]         # per-material keys allowed, e.g. TiO2 = [10, 300]

[back]                         # optional, plate modes only; optimize = false needs thickness_nm
materials = ["FusedSilica", "TiO2"]

[[targets]]                    # one row per requirement
name = "pass"                  # unique
beam = "pass"                  # operating condition label (default: name); rows on one beam share it
side = "front"                 # front | back (back face, or from the substrate side in the stack modes)
wavelength_nm = [520, 545]     # scalar, explicit list, or [start, stop] with points = N
points = 26
aoi_deg = 0.0                  # scalar or [start, stop] with aoi_points = N; in the incidence medium
polarization = "unpolarized"   # s | p | unpolarized | both | azimuth in degrees (0 = p, 90 = s)
quantity = "T"                 # R | T | A
kind = "min"                   # eq | min (lower bound) | max (upper bound)
value = 0.98
tolerance = 1.0                # deviation is divided by this
weight = 1.0                   # per row, split over its expanded points

[objective]
form = "power"                 # power: MF = (sum w |d|^p / sum w)^(1/p) | minimax | log
power = 2

[[report.sums]]                # optional, e.g. output ports
name = "left port"
terms = ["T@1", "R@2"]         # quantity@beam (single-point beams)

[optimizer]                    # n_starts, adam_steps, adam_lr, top_k, seed, global_de, de_maxiter, de_popsize

[[materials.custom]]           # type = constant (n, k) | sellmeier (B, C) | csv (path: wavelength_nm,n,k)
```

| Substrate mode | Stack | Use |
|---|---|---|
| `incoherent` | ambient \| front \| thick substrate \| back \| ambient | real plates; reports the fringe envelope and back-face ceiling |
| `coherent` | same, with the substrate as an etalon at its physical phase | single-frequency analysis |
| `semi_infinite` | ambient \| front \| substrate | substrate "off": back surface ignored, no fringes |
| `none` | ambient \| front \| exit medium | free-standing stack |

## Examples

| Config | What it shows | Result (`results/<config>/`) |
|---|---|---|
| `combiner_45.toml` | 45° combiner, maximize T1 and R2, bare back face | T1 = 0.917358 (bare-face ceiling), R2 = 0.999629 |
| `combiner_45_up_port.toml` | maximize R1 and T2 | R1 = 0.999839, T2 = 0.919968 |
| `combiner_45_back_ar.toml` | plus an optimized 4-layer back-face AR | T1 = 0.999998, R2 = 0.999730 |
| `combiner_45_back_ar_sp.toml` | targets for s and p (`polarization = "both"`) | T1 ≥ 0.9972, R2 ≥ 0.9924 |
| `normal_dichroic_bands.toml` | 0°, wavelength bands, one-sided targets, semi-infinite substrate | pass band T ≥ 0.9892, reflect band R ≥ 0.99975 |

Custom nk tables go in `materials/<Name>.csv`, with the columns `wavelength_nm,n,k`.

## Layout

| Path | Purpose |
|---|---|
| `src/ifopt/targets.py` | Target rows, expansion, beams, deviations |
| `src/ifopt/config.py` | TOML schema → dataclasses, validation, design vector |
| `src/ifopt/plate.py` | Substrate modes, evaluation `Model`, fringe scan, back-face ceiling, engine cross-check |
| `src/ifopt/engines/` | S-matrix (default), tmmax adapter, Byrnes reference |
| `src/ifopt/objectives.py`, `optimize.py` | Merit forms, quarter-wave seed, vmapped Adam → L-BFGS-B, optional DE |
| `src/ifopt/analysis.py` | Per-beam spectra vs wavelength and angle, thickness Monte Carlo, plots |
| `src/ifopt/materials.py` | Material library; raises outside the tabulated range |
| `src/ifopt/geometry.py` | `system.xml` port mapping (combiner tests) |
