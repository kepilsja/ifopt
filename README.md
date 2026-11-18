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

## Python API

```python
import ifopt as fo

b1 = fo.Beam("1", wavelength_nm=532, aoi_deg=45, side="front", polarization="s")
b2 = fo.Beam("2", wavelength_nm=1064, aoi_deg=45, side="back", polarization="s")
system = fo.FilterSystem(
    substrate=fo.Substrate("FusedSilica", mode="incoherent", thickness_mm=1.0),
    front=fo.Coating(fo.alternating("TiO2", "FusedSilica", 15), bounds_nm={"default": (5, 400)}),
    targets=[b1.eq("T", 1.0, name="T1"), b2.eq("R", 1.0, name="R2")],
)                                                   # or fo.FilterSystem.from_toml("config/combiner_45.toml")

ev = fo.evaluate(system, fo.Design.quarter_wave(system))     # Evaluation: beams, targets, merit, sums
opt = fo.optimize(system, settings=fo.OptimizerSettings(n_starts=64))
opt.design.save("best.json", system)
variant = system.replace(substrate=fo.Substrate("FusedSilica", mode="semi_infinite"))   # validated copy
spec = fo.spectrum(system, opt.design, wavelengths_nm=range(450, 1201), aoi_deg=45)
fig = fo.plot_spectra(system, opt.design)                    # matplotlib Figure
system.to_toml("my_system.toml")
```

| Object / function | Purpose |
|---|---|
| `FilterSystem`, `Substrate`, `Coating`, `Target`, `Beam`, `Objective`, `OptimizerSettings`, `ReportSum` | Frozen spec objects, validated at construction. Vary them with `.replace(...)`. Use `from_toml`/`from_dict` to load and `to_toml`/`to_dict` to save. |
| `MaterialLibrary` (`add_constant`, `add_sellmeier`, `add_table`, `add_csv`), `ConstantMaterial`, ... | Materials. A `Material` object can also be placed directly in a `Coating`, and it is registered automatically. |
| `Design` (`quarter_wave`, `from_system`, `from_x`, `x`, `save`, `load`) | Layer thicknesses of the front coating (and back coating, if any). |
| `evaluate` → `Evaluation` | R/T/A per beam, target results, merit/MF, sums, fringe envelope, ceiling, engine cross-check. Also `.report()`, `.to_dict()`, `.to_pandas()`. |
| `optimize` → `OptimizationResult` | Multi-start Adam followed by L-BFGS-B. Pass `start=` for extra starting designs and `settings=` to override the optimizer settings. |
| `spectrum`, `beam_spectrum`, `angle_scan` → `Spectrum` | Sweeps on any wavelength or AOI grid. |
| `monte_carlo` → `MonteCarlo` | Robustness to thickness errors. |
| `stack_rt` | R/T of a hand-written stack, without building a system. |
| `plot_spectra`, `plot_stack`, `plot_convergence`, `save_plots` | Figures. |

**Library behavior:**
- **Logging:** ifopt never prints. It logs through `logging.getLogger("ifopt")` and is silent by default. Call `fo.enable_logging()` to see optimizer progress.
- **Precision:** importing ifopt turns on JAX 64-bit mode. API calls also force it while they run, so results are always float64 NumPy arrays.

The scripts in `examples/` show the API in use:
- `scripted_combiner.py` builds the combiner in code and checks that it equals the TOML config;
- `parameter_sweep.py` re-optimizes the bands filter over AOI and substrate models and prints a pandas table;
- `quick_stack.py` evaluates a hand-written quarter-wave mirror.

## Command line

```bash
.venv/bin/ifopt evaluate --config config/combiner_45.toml
```

```bash
.venv/bin/ifopt optimize --config config/combiner_45.toml --out results/combiner_45
```

```bash
.venv/bin/ifopt optimize --config config/normal_dichroic_bands.toml --out results/normal_dichroic_bands
```

```bash
.venv/bin/ifopt evaluate --config config/combiner_45.toml --design results/combiner_45/design.json
```

```bash
.venv/bin/ifopt stability --config config/combiner_45.toml
```

```bash
.venv/bin/python -m pytest -q
```

- **Logging options:** `-v` (debug), `-q` (warnings only) and `--log-file PATH` work on every command. `python -m ifopt ...` is equivalent to `ifopt`.
- **Old scripts:** `scripts/compute_rt.py`, `optimize_filter.py` and `stability_study.py` remain as shims for `ifopt evaluate`, `ifopt optimize` and `ifopt stability`.
- **`evaluate`** works on a saved `design.json`, the thicknesses given in the config, or the quarter-wave seed. It prints R/T/A per beam, the engine differences, the fringe envelope and ceiling, the target table, the merit and the sums.
- **`optimize`** writes `design.json`, `report.txt` (with a thickness Monte Carlo) and the plots. `--starts`, `--steps` and `--global-de` override the optimizer settings.

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
| `src/ifopt/api.py` | Public operations and result types (`evaluate`, `optimize`, `spectrum`, `monte_carlo`, `stack_rt`, report) |
| `src/ifopt/cli.py` | `ifopt` command (`evaluate`, `optimize`, `stability`), logging setup |
| `src/ifopt/design.py`, `io.py` | `Design` (thicknesses, design.json); TOML writer |
| `src/ifopt/targets.py` | `Target`, `Beam`, resolution, expansion, deviations |
| `src/ifopt/config.py` | `FilterSystem` and spec objects, validation, dict/TOML round trip, design vector |
| `src/ifopt/plate.py` | Substrate modes, evaluation `Model`, fringe scan, back-face ceiling, engine cross-check |
| `src/ifopt/engines/` | S-matrix (default), tmmax adapter, Byrnes reference |
| `src/ifopt/objectives.py`, `optimizer.py` | Merit forms, quarter-wave seed, vmapped Adam → L-BFGS-B, optional DE |
| `src/ifopt/analysis.py`, `plotting.py` | Spectra and Monte Carlo computations; Figure-returning plots |
| `src/ifopt/stability.py` | Numerical-stability study (`ifopt stability`) |
| `src/ifopt/materials.py` | Material library; raises outside the tabulated range |
| `src/ifopt/geometry.py` | `system.xml` port mapping (combiner tests) |
