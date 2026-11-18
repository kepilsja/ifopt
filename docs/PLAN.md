# Plan: Two-wavelength 45° dichroic plate — R/T calculation and optimization (Python 3.14 + tmmax)

> **Status:** implemented. Section 12 lists what was *measured* during implementation; where it differs from the
> predictions in sections 3–5, section 12 is authoritative.

## 1. Context

`prompt.md` asks for two things, built in Python 3.14 around the `tmmax` package:
1. Scripts that compute the transmittance and reflectance of one multilayer filter plate for two monochromatic sources.
2. An optimizer that maximizes a chosen pair of coefficients, for example T1 and R2.

It also asks for an analysis of what tmmax can do, where it is weak (especially numerical stability with highly lossy layers), and alternatives with pros and cons for each weak point.

The project folder currently holds only `prompt.md` and `system.xml`. There is no existing code to reuse.

**Decisions taken (all were the recommended options):**
- **Hybrid engine.** tmmax is used for baseline R/T, nk data and cross-checks. A small float64 JAX S-matrix engine of our own handles optimization and lossy stacks. Both sit behind one interface.
- **Plate model.** A thick substrate with the coating on one face. The substrate is treated incoherently, and a coherent fringe envelope is reported alongside.
- **Optimizer scope.** Layer thicknesses only, with a fixed material sequence. Multi-start gradient search followed by an L-BFGS-B polish. Robustness terms and needle synthesis come later.

**Environment (checked on this machine).**
- Python 3.14.7 (arm64) is at `/opt/homebrew/Caskroom/miniconda/base/bin/python3.14`.
- None of jax, jaxlib or tmmax is installed yet.
- PyPI offers cp314/arm64 wheels for jax and jaxlib 0.7.1 through 0.12.0.1.
- The latest tmmax on PyPI is 1.1.4 (Oct 2025).

## 2. System decoded from `system.xml`

The plate is the line from (364,480) to (464,580) in the drawing, so it sits at 45°. The two sources hit **opposite faces** of it:

| Beam | Arrives | Transmitted goes | Reflected goes |
|---|---|---|---|
| λ1 (red) | from the right, onto the upper-right face | **T1 → left** | **R1 → up** |
| λ2 (green) | from below, onto the lower-left face | **T2 → up** | **R2 → left** |

This makes the plate a **dichroic beam combiner**. The left port carries {T1, R2} and the up port carries {R1, T2}, so "maximize T1 and R2" means "combine both beams into the left port".

Consequences for the model:
- **λ2 crosses the layers in reverse order** (substrate side first, if the coating faces λ1). The code therefore needs R and T from both sides of the stack.
  - With absorption present, R_front ≠ R_back.
  - T_front = T_back always holds (reciprocity), which gives a free test.
- **Both beams travel in the drawing plane**, which is therefore the plane of incidence for both. "s" for both sources means polarization perpendicular to the drawing.
  - At 45°, s and p differ strongly, so polarization is set per source: s, p, unpolarized, or a linear azimuth ψ.
  - For isotropic stacks, mixing is exact: P = cos²ψ·P_s + sin²ψ·P_p.

## 3. What tmmax offers, and how we use it

Findings come from the README, the docs and a reading of the master-branch source (`tmm.py`, `angle.py`, `fresnel.py`, `wavevector.py`, `cascaded_matmul.py`, `reflect_transmit.py`, `data.py`).

**Public API.**
```python
from tmmax.tmm import tmm
R, T = tmm(material_list, thickness_list, wavelength_arr, angle_of_incidences,
           coherency_list=None, polarization='s')
```
- `material_list` is a list of strings: incident medium, the layers, then the exit medium.
- Thicknesses and wavelengths are in metres. Angles are in radians.
- The output is indexed `[angle, wavelength]`. It is vmapped over wavelength and angle, and the cascade runs in `lax.scan`.
- **Useful to us:**
  - The material database: about 30 dielectrics, including TiO2, SiO2, Ta2O5, Nb2O5, HfO2, MgF2, Al2O3 and ZnS, plus Si and Ge. It has no metals.
  - Coherent R and T.
  - The `analysis.thickness_sensitivity` Monte-Carlo tool.
  - The internal jitted `vectorized_coh_tmm(data, material_distribution, thickness_list, wavelengths, angles, polarization)`, which can be differentiated.

**How tmmax is used here.** It sits behind an adapter (`engines/tmmax_engine.py`) that does three things:
1. Builds the `data` array in memory from our own material library, so nothing is written into site-packages.
2. Calls `vectorized_coh_tmm` for coherent R and T. For λ2 it calls it once more with the stack reversed, giving R_a′ and T_a′.
3. Hands those values to our own closed-form incoherent-substrate formula (§6). tmmax's incoherent code path is not used.

This gives a full independent second engine for the whole plate. Inside its single-precision range, it serves as the baseline and cross-check.

## 4. Numerical stability with lossy materials — analysis

**Mechanism.**
- Each layer's phase is δ = 2π·d·q/λ with q = √(n² − n₀²sin²θ₀).
- An absorbing layer has Im q > 0, so in the forward branch |e^{−iδ}| = e^{Im δ} **grows**.
- tmmax builds the Byrnes-style transfer matrix `(1/t)·[[e^{−iδ}, r·e^{−iδ}], [r·e^{iδ}, e^{iδ}]]` and multiplies it through the stack without normalization. So the size of the entries grows like exp(Σ Im δ_j).
- There is no capping of Im δ. Byrnes' `tmm` package caps it at about 35; tmmax does not.

**tmmax hard-codes `complex64`** (`coh_matmul`, `jnp.eye(2, dtype=complex64)`, `create_nk_list(...).astype('complex64')`). Setting `jax_enable_x64` does not change this.

| Limit | float32 / complex64 (tmmax) | float64 / complex128 |
|---|---|---|
| Overflow of e^{Σ Im δ} | Σ Im δ ≳ **88.7** | Σ Im δ ≳ **709.8** |
| Gradient path (chain rule forms 1/M₀₀², which overflows at half the exponent) | Σ Im δ ≳ ~44 | Σ Im δ ≳ ~355 |
| Objective resolution (eps) | ~1.2·10⁻⁷ | ~2.2·10⁻¹⁶ |

Rough worked numbers (to be confirmed in Phase 2):

| Case | Im δ per nm | Overflow in float32 | Overflow in float64 |
|---|---|---|---|
| Ag at 532 nm, 45° (q ≈ 0.05+3.5i) | ≈ 0.041 | ~2.1 µm of Ag (~1 µm for gradients) | ~17 µm |
| Al at 532 nm (q ≈ 0.95+6.7i) | ≈ 0.080 | ~1.1 µm | ~9 µm |
| 0.5 mm Si wafer at 532 nm, treated coherently | Im δ ≈ 240 | overflows | finite, but physically meaningless (should be incoherent) |

**What this means in practice.**
- **Thin metal layers (under ~100 nm) evaluate fine even in tmmax.**
- The real risks are elsewhere:
  - (a) thick absorbing layers, or an absorbing substrate treated coherently;
  - (b) an optimizer that drifts a lossy layer thick and then hits inf or NaN, which aborts the run;
  - (c) gradient overflow, which happens before forward overflow;
  - (d) the float32 objective noise floor of about 10⁻⁷, which stalls L-BFGS line searches near the optimum;
  - (e) A = 1 − R − T coming out slightly negative.
- **One-sided r and t stay accurate up to the point of overflow.** They are ratios of the dominant terms. Precision is lost only in sub-dominant terms, such as the contribution of layers behind an opaque absorber, which is physically negligible.

**Other numerical weak spots found in tmmax.**
- `R = square(abs(r))`. The gradient of abs is NaN at r = 0.
- `jnp.select` evaluates both the s and p branches, so a singular result in the unused branch can leak NaN into gradients.
- The arcsin-based angles have an infinite derivative at sin = 1.
- The forward-branch correction for complex angles is **inverted, and in effect never runs**. This is harmless here because air is both the incident and the exit medium, but it is wrong for an absorbing incident or exit medium.
- **The incoherent path** (`coherency_list` with 1 = thick layer). It follows Katsidis & Siapkas: each coherent sub-stack is reduced to intensities (R, T from both sides), and each thick layer contributes an intensity propagation matrix diag(1/p, p), where p = |e^{iδ}|² = exp(−2 Im δ) is its single-pass internal transmittance. Three problems, all found by reading the source (not yet run — Phase 0 confirms them):
  1. **Runs eagerly in Python, without JIT.**
     - `vectorized_incoh_tmm` has no `@jit`.
     - The core helper `compute_r_t_magnitudes_incoh` is defined three times in `reflect_transmit.py`. The last definition wins, and the middle one calls a helper with the wrong number of arguments.
     - Code comments say it is "not good for jit". It loops in Python with `if condition:` branches.
     - Consequences: it is slow (op-by-op dispatch on every call), and a Python `if` on a traced value raises an error under `jax.jit` or `jax.grad`. So it cannot sit inside a gradient-based optimizer.
  2. **Probable off-by-one in which layer's absorption is applied.**
     - The code mixes two numberings.
       - *nk numbering:* 0 = ambient, 1…N = layers, N+1 = exit medium.
       - *Layer numbering:* 0…N−1. This is the numbering of `thickness_list`, `coherency_list` and `layer_phases = kz[1:-1]·thickness_list`.
     - `find_coh_and_incoh_indices` pads `coherency_list` with a 1 at each end, so the indices it returns are in nk numbering. `tmm.py` then strips the ends with `.at[1:-1]` and passes the indices on unchanged: `compute_inc_layer_pass(incoherent_layer_indices_forward.at[1:-1].get(), layer_phases)`. That function then does `jnp.take(layer_phases, idx)`, which expects layer numbering.
     - README example: `coherency_list = [1,0,1,0,1,0]` (2000 nm thick layers alternating with 100 nm coherent ones).
       - Padded, this becomes `[1,1,0,1,0,1,0,1]`, so the incoherent indices are [0,1,3,5,7]. After `[1:-1]` they are [1,3,5].
       - The code therefore reads the phases of the **100 nm coherent layers 1, 3, 5** instead of the 2000 nm thick layers 0, 2, 4.
     - Docstring example `[0,0,0,1,0,0,1]`: the code ends up asking for index 7 of a 7-element array.
       - JAX's default out-of-bounds mode for `take` returns NaN for float types.
     - **The bug is invisible for lossless materials.** With Im δ = 0, p = 1 whichever layer is read. It only shows up with absorption, which is exactly the case this project cares about.
  3. **`1/p` overflow, and p underflow.**
     - The intensity matrix contains 1/p = exp(+2 Im δ), the squared version of the amplitude problem in the table above.
     - In float32, 1/p overflows once 2 Im δ > 88.7. For 1 mm at 532 nm that happens at k ≳ 3.8·10⁻³, i.e. once the layer is effectively opaque (internal transmittance below about 10⁻³⁸).
     - At that point p underflows to 0 and 1/p becomes inf. The next products are inf·0 = NaN.
     - The physically correct answer is T = 0 and R = R_front. tmmax returns NaN instead.
     - Realistic triggers: a Si or Ge wafer, or absorbing filter glass, used in its opaque band.
     - The incoherent cascade is also float32 (`jnp.eye(2, dtype=float32)`).
  - **Our replacement:** the closed-form formula in §6. τ appears only multiplied, never divided by, so an opaque layer goes smoothly to T → 0 and R → R_a. It also needs only one thick layer, which is all this plate has.

## 5. Weak points and alternatives (★ = chosen)

| # | Weak point | Alternatives | Pros | Cons |
|---|---|---|---|---|
| W1 | complex64 hard-coded | ★ A: own float64 engine | full precision, independent of tmmax | more code to validate |
| | | B: fork tmmax with dtype following the x64 flag | small diff that can go upstream | a fork to maintain; W2, W4 and W5 remain |
| | | C: accept float32 | no work | 10⁻⁷ noise floor, early overflow |
| W2 | Overflow in the T-matrix cascade | ★ A: **S-matrix / Redheffer star product** (only e^{+iδ} terms, \|·\| ≤ 1) | unconditionally stable; r, t, r′, t′ from both sides in one pass (needed for λ2 and the substrate); associative | about 2–3× the arithmetic per layer (irrelevant at ~50 layers) |
| | | B: Rouard / Airy reflection recursion | simplest, stable, scalar | one direction per pass; no field profiles |
| | | C: cap Im δ (Byrnes "opacity" trick) | one line | zero gradient past the cap; an approximation |
| | | D: renormalize the cascade with a log-scale | small change | fixes range, not conditioning; awkward under autodiff |
| | | E: float64 alone | 8× more headroom | only moves the wall; Apple-GPU Metal backend has no float64 (CPU is fine at this size) |
| W3 | `tmm()` cannot be differentiated (`np.any` on traced inputs) and reads `.npy` files on every call | ★ A: own engine for optimization, plus tmmax's internal `vectorized_coh_tmm` for cross-checks | gradients and JIT on both paths | the tmmax entry point is private API, so pin the version |
| | | B: gradient-free search over the public `tmm()` | uses public API only | slow; poor in 20+ dimensions |
| W4 | Incoherent path (eager, off-by-one, `1/pass`, float32) | ★ A: own closed-form thick-substrate formula built on coherent R and T | exact for a single thick layer; τ ≤ 1 is never divided by | covers only a single thick layer (that is all this system has) |
| | | B: fix it upstream | helps the community | depends on someone else's timeline |
| | | C: treat the substrate coherently | exact for an ideal parallel plate | unrealistic fringes; overflows with absorption |
| W5 | Branch of the complex angle | ★ work with kₓ = n₀ sin θ₀ and q = √(n² − kₓ²), forcing Im q ≥ 0 (and Re q ≥ 0 when Im q = 0) | no arcsin; stable; correct for lossy media | none to speak of |
| W6 | Materials: names only; linear interpolation that silently clamps outside the table; µm in the DB but metres in the API; `add_material_to_nk_database` clips k at 20 and writes into site-packages; no metals | ★ own `MaterialLibrary`: reads tmmax `.npy` files read-only, plus user CSVs (e.g. refractiveindex.info Ag/Al/Au/Cr), plus constants; raises outside the tabulated range | reproducible; supports metals | a small amount of code |
| | | alt: the `refractiveindex` PyPI package | very large database | extra dependency, downloads the DB |
| W7 | NaN gradients | ★ R = Re(r·r̄); "double-where" guards; bounded parametrization | smooth gradients | needs some care |
| W8 | Versions: PyPI 1.1.4 (Oct 2025) lacks the Feb-2026 "matmul bug fix" on master; tmmax was written against jax 0.5, but Python 3.14 needs jax ≥ 0.7.1 | ★ Phase-0 smoke test of PyPI 1.1.4 against master pinned by commit; pin whichever passes | reproducible | — |
| W9 | Precisely defined wavelengths mean long coherence, so a real 1 mm plate shows etalon fringes | ★ incoherent result (equal to the phase average) plus a coherent min/max envelope over substrate phase φ ∈ [0, 2π) | shows the fringe risk; recommend a wedge or back-side AR coating | — |
| W10 | JIT overhead for tiny problems | jit once with fixed shapes; batch the multi-start with vmap | — | the first call takes about a second to compile |

## 6. Physics model (as implemented)

- **Units.** Thickness and wavelength are in **nm** throughout the engine and the optimizer, which keeps the optimizer well conditioned. The tmmax adapter converts to metres.
- **Coherent stack (`engines/stable_engine.py`).**
  - Uses the Byrnes conventions (arXiv:1603.02720), the same as tmmax and the `tmm` package, so the three engines can be compared term by term.
  - Each interface's S-matrix comes from the Fresnel r and t written in terms of q (s) or n²/q (p).
  - Each layer is propagated with t = t′ = e^{iδ}.
  - The layers are combined with the star product inside `lax.scan`.
  - Returns r, t, r′, t′ and the powers R, T, R′, T′. T uses the Poynting normalization Re(q_out)/Re(q_in) and its p-polarization equivalent.
  - Materials are an integer-index array into a complex index table of shape `[n_materials, n_wavelengths]`. Shapes are fixed, so nothing recompiles during optimization.
- **Plate (`plate.py`).** The stack is ambient | coating on face F | substrate (thickness D) | back face | ambient. F is set by `coated_face`. The back face is bare (or AR later).
  - For a source arriving on face X:
    - a is the stack on X, with R_a, T_a seen from outside and R_a′, T_a′ seen from inside;
    - b is the stack on the opposite face, seen from inside;
    - τ = exp(−4π·D·Im q_s/λ) is the single-pass intensity transmission of the substrate.
  - The combination formulas:
    - `R = R_a + T_a·T_a′·R_b·τ² / (1 − R_a′·R_b·τ²)`
    - `T = T_a·T_b·τ / (1 − R_a′·R_b·τ²)`
    - `A = 1 − R − T`
  - These are applied **per polarization** and the results are mixed afterwards.
  - **Coherent plate vs incoherent plate.** Light bounces back and forth inside the ~1 mm substrate between its two faces.
    - *Coherent plate:* the substrate is treated like one more thin film. The **complex amplitudes** of all the bounced beams are added, phase included. The plate then behaves as a Fabry–Pérot etalon.
      - R and T oscillate with substrate round-trip phase φ = 2π·2·d_s·Re(q_s)/λ.
      - For 1 mm of glass at 532 nm the fringe period is only Δλ ≈ λ²/(2·n·d·cosθ) ≈ 0.1 nm. Tiny changes in temperature, plate thickness or angle shift it by a whole fringe.
      - Formula: `T = T_a·T_b·τ / |1 − r_a′·r_b·τ·e^{2iφ}|²`.
    - *Incoherent plate:* **intensities** of the bounced beams are added and phase is ignored. This is the §6 formula: smooth, with no fringes.
    - *Which is real?*
      - Broadband or short-coherence sources: incoherent.
      - Single-frequency lasers ("precisely defined wavelengths", coherence length up to metres) on a truly parallel plate: the fringes physically exist. In practice they are partly washed out by thickness non-uniformity across the beam, and fully removed by a wedged substrate.
    - *Mathematical link:* averaging the coherent result over φ ∈ [0, 2π) gives exactly the incoherent result (Parseval). This becomes a unit test.
    - *Fringe envelope (reported):* the coherent plate evaluated for all φ (vmapped), with the min and max of each coefficient reported.
      - Peak-to-peak swing in T is (1 ± √(R_a′R_b))⁻².
      - Example: residual coating reflectance R_a′ = 1% and a bare back face (s-pol at 45°, n = 1.5, so R_b ≈ 9.2%). Then √(R_a′R_b) ≈ 0.03 and T1 swings by about **±6%**. The incoherent value differs from the no-back-face value by only 0.1%.
      - This is how the tool flags when a back-side AR coating or a wedge is needed.
- **Output** for each source: T, R and A for s, p and the mixed polarization, plus the port totals (left = T1 + R2, up = R1 + T2).

## 7. Optimization design (thickness-only)

- **Variables.** The thicknesses d_j of a fixed material sequence given in config, for example `[TiO2, SiO2] × N` with optional thin metal layers. Each layer has bounds by material, e.g. dielectrics in [5, 500] nm and metals in [0, 40] nm.
- **Merit function** (minimized). The selected coefficients come from config, e.g. `maximize = ["T1","R2"]` or `port = "left"`.
  - Default least squares to target 1: `L = Σ w_k (1 − P_k)²`.
  - Optional forms: `log` (−Σ w log P_k, which balances the pair) and `softmin` (smooth max-min).
  - Option `both_polarizations = true`, which applies the targets to s and p separately.
- **Pipeline.**
  1. **Seeds.** The analytic seed is a quarter-wave (HL)ⁿH stack for the reflected wavelength at 45°, with d_j = λ/(4·Re q_j). It is joined by 256–1024 uniform random starts.
  2. **Stage A.** optax Adam over the unconstrained u, with d = d_min + (d_max − d_min)·sigmoid(u). All starts are vmapped in one jitted call for about 500 steps.
  3. **Stage B.** The top k designs are polished with `scipy.optimize.minimize(method="L-BFGS-B", jac=True, bounds=...)` on `jax.value_and_grad`, in float64.
     - *What L-BFGS-B is:* a quasi-Newton local optimizer. It estimates the curvature of the merit from the last ~10 gradient differences (that is the "Limited-memory BFGS" part). It enforces box bounds d_min ≤ d_j ≤ d_max natively (that is the "-B"). It converges much faster than Adam near an optimum, but it is local, so it is used only to polish the best multi-start results.
     - Its line search compares small changes in the merit, so it needs a float64 objective and gradient.
     - *Alternative:* `scipy.optimize.least_squares(method="trf", bounds=...)`, which is Levenberg–Marquardt-style damped least squares, the classic choice in thin-film design software. It fits our least-squares merit and takes its Jacobian from `jax.jacfwd`.
  - **Global alternative:** `scipy.optimize.differential_evolution(vectorized=True)` evaluating the vmapped objective, then the same polish.
  4. **Stage C.** Re-evaluate the best design with tmmax and with the reference `tmm` package. Report all six coefficients plus A, the fringe envelope, and a thickness Monte Carlo (tmmax `thickness_sensitivity`, or our own).
  5. Plot R and T against λ (±50 nm around each line) and against AOI (40–50°).
  - Save the design as JSON.

## 8. Project structure (new files)

```
pyproject.toml                 # requires-python >=3.14; jax>=0.7.1, optax, scipy, numpy, matplotlib,
                               # tmmax (pinned; PyPI or git commit per Phase 0), tmm (reference, tests), pytest
config/example.toml            # wavelengths, AOI=45, polarization per source, ambient/substrate,
                               # coated_face, layer sequence + bounds, objective, optimizer settings
materials/*.csv                # user nk tables (wavelength_nm,n,k), e.g. Ag/Al from refractiveindex.info
src/ifopt/
  config.py                    # tomllib -> frozen dataclasses + validation
  geometry.py                  # system.xml mapping: source -> face, coefficient -> output port
  materials.py                 # MaterialLibrary (tmmax .npy read-only, CSV, constants), range checks
  engines/base.py              # Engine protocol: stack_rt(idx, d_nm, nk_table, kx, lam, pol) -> r,t,r',t',R,T,R',T'
  engines/stable_engine.py     # JAX complex128 S-matrix (default)
  engines/tmmax_engine.py      # adapter around tmmax vectorized_coh_tmm (+ reversed-stack call)
  engines/reference_engine.py  # Byrnes `tmm` (numpy float64), used for tests only
  plate.py                     # incoherent substrate combination, polarization mixing, fringe envelope
  objectives.py                # least_squares / log / softmin merits over the selected coefficients
  optimize.py                  # parametrization, vmapped Adam multi-start, L-BFGS-B polish, optional DE
  analysis.py                  # spectra vs λ/AOI, sensitivity, stability diagnostics (Σ Im δ, dtype)
  plotting.py
scripts/compute_rt.py          # Step 1 CLI: table of T/R/A for both sources, both engines, plus differences
scripts/optimize_filter.py     # Step 2 CLI: --maximize T1 R2 | --port left; writes results/<run>.json + plots
scripts/stability_study.py     # sweeps lossy-layer thickness; shows where tmmax overflows, stable engine stays finite
tests/                         # see Verification
docs/PLAN.md                   # copy of this plan
```

## 9. Implementation phases

0. **Environment and smoke test.**
   - Create the venv: `python3.14 -m venv .venv`, then install the dependencies.
   - Verify:
     - tmmax imports and runs its README example under jax 0.12;
     - the output dtype is complex64/float32;
     - the real signature of `vectorized_coh_tmm` (and how polarization is encoded);
     - that `jax.grad` through `tmm()` fails as predicted;
     - the incoherent-path findings (§4), checked against the analytic formula:
       - a thick absorbing layer next to a lossless thin layer, to expose the off-by-one;
       - the same thick layer with a lossless neighbour of different thickness, to show which layer's phase is actually read;
       - an opaque thick layer, to show the NaN;
     - PyPI 1.1.4 against the master commit.
   - Pin the versions. Copy this plan to `docs/PLAN.md`.
1. **Step 1, calculation.** Write `config`, `materials`, `geometry`, `stable_engine`, `tmmax_engine`, `plate` and `scripts/compute_rt.py`.
2. **Validation and stability study.** Write the tests and `scripts/stability_study.py`. Record the measured overflow thresholds in `docs/PLAN.md` §4.
3. **Step 2, optimization.** Write `objectives`, `optimize`, `analysis`, `plotting` and `scripts/optimize_filter.py`. Demonstrate it on the example config, maximizing T1 + R2 (left port).
4. **Later, out of scope now.** Robustness-averaged merit (AOI and λ spread, thickness errors), needle synthesis (the gradient with respect to a zero-thickness inserted layer gives the needle function directly through JAX), and co-optimization of a back-face AR coating.

## 10. Verification

`pytest`:
- **Analytic checks.**
  - A single interface at 45° (s and p) against the closed-form Fresnel equations.
  - p-reflectance is 0 at Brewster's angle.
  - A quarter-wave stack matches the closed-form peak R.
  - A single absorbing film matches the Airy formula.
- **Invariants.**
  - R + T = 1 for lossless stacks, and A ≥ 0 always.
  - T_front = T_back (reciprocity, including lossy stacks).
  - R_front = R_back for lossless stacks, and R_front ≠ R_back for lossy ones.
  - s = p at normal incidence.
- **Cross-engine comparison** on random dielectric and thin-metal stacks:
  - stable engine against `tmm` (Byrnes): rtol 1e-10;
  - stable engine against tmmax: rtol about 1e-5 (float32).
- **Substrate.** The coherent plate averaged over substrate phase equals the incoherent formula.
- **Gradients.** `jax.test_util.check_grads` and central differences in float64.
- **Lossy stress test.** Sweep Ag from 10 nm to 30 µm.
  - The stable engine stays finite, T decreases monotonically to 0, and R tends to bulk Ag.
  - tmmax produces inf or NaN past the threshold predicted in §4.

End-to-end runs:
- `python scripts/compute_rt.py --config config/example.toml` prints T1, R1, A1, T2, R2, A2 (s, p and mixed) for both engines, with their difference.
- `python scripts/optimize_filter.py --config config/example.toml --maximize T1 R2` reports the best design and its merit. It also confirms the result with the tmmax and `tmm` engines and saves the JSON and plots.
- `python scripts/stability_study.py` produces the overflow plot.

## 11. Inputs the user supplies in config (placeholders until then)

- The two wavelengths. The example uses 532 and 1064 nm as placeholders.
- The polarization of each source.
- The substrate material and thickness, and which face carries the coating.
- The candidate coating materials and the layer template.
- Any metal nk tables (CSV), since tmmax's database has no metals.

## 12. Measured findings (implementation, 2026-11-18)

Environment: Python 3.14.7 (arm64), jax/jaxlib 0.12.0.1, optax 0.2.9, tmm 0.2.0, tmmax pinned to master commit
`dc4b904` (see below). Evidence: `scripts/stability_study.py` → `results/stability/stability_report.md`, plus `pytest`.

**New problems with tmmax (not anticipated in §4):**
- **PyPI tmmax 1.1.4 gives wrong R/T for any stack with 2 or more layers.** `coh_matmul` builds a (2,2,1) array, and
  `jnp.matmul(carry, M)` then computes M·carryᵀ instead of carry·M. For example, for 4 layers at 532 nm (s-pol),
  tmmax gives R = 0.529 where the reference gives 0.767. GitHub master (Feb 2026 "matmul bug fix") is correct,
  matching to about 1e-6, so `pyproject.toml` pins commit `dc4b90438dffeafd420a7b90c63539813881bedb`.
- **tmmax crashes as soon as `jax_enable_x64` is on**, with `TypeError`: the scan carry is complex64 but the
  output is complex128. The adapter therefore runs every tmmax call inside `jax.enable_x64(False)`.
- **A bare interface (zero layers) crashes** with an `IndexError` after `jnp.squeeze`. The adapter inserts a
  zero-thickness, index-matched layer instead.
- **nk database problems** (audited at 532 and 1064 nm):
  - `SiO2` has k = 2.0e-3 at 532 nm, which would make a 1 mm plate opaque (τ ≈ e⁻⁴⁷). Hence the built-in
    `FusedSilica` Sellmeier.
  - `Si` covers only 2–20 µm, and the `Kischkat_*` tables only 1.5–14 µm. tmmax silently clamps to the end of
    the table.
  - `CdS` has negative k (which would mean gain).
  - `ZnS`, `Ta2O5`, `SiO`, `CeF3` and `ITO` have k > 1e-3.

**Corrections to the predictions:**
- `jax.grad(tmm(...))` works when run eagerly. Only `jax.jit(jax.grad(tmm(...)))` fails, with
  `TracerBoolConversionError`.
- For an opaque incoherent layer, tmmax returns a **wrong finite value** (T = 0.30, where the reference gives
  2e-31), not NaN. The off-by-one reads the lossless neighbouring layer, so 1/p never overflows.

**Overflow thresholds:** single absorbing layer on fused silica, 45°, s-pol, thickness sweep 10 nm–60 µm.

| Material @ 532 nm | tmmax R/T → NaN | tmmax gradient wrong | complex128 T-matrix fails | S-matrix |
|---|---|---|---|---|
| Ge (tmmax DB, q = 3.42+2.61i) | 2.9 µm (Σ Im δ = 89.5) | 1.5 µm (Σ Im δ = 46) | 24.8 µm (Σ Im δ = 765) | finite to 60 µm, |ΔR| vs Byrnes ≤ 6e-16 |
| metal-like 0.05+3.4i | 2.3 µm (96) | 1.1 µm (46) | 18.4 µm (757) | finite, ≤ 9e-16 |

These confirm the predicted thresholds (88.7, about 44 and 709.8; the measured values are higher only because the
thickness grid is coarse). In addition, the float32 gradient sits on a noise floor of about 1e-8/nm from roughly
0.2 µm onward. In the incoherent path, tmmax gives T = 0.441 for a 2 µm Ge (incoherent) + 100 nm TiO2 stack at
1064 nm, while Byrnes `inc_tmm` and our closed form both give 0.0011.

**Example results** (`config/example.toml`: 532 + 1064 nm, s-pol, fused silica 1 mm, 15-layer TiO2/SiO2 coating
facing source 1, bare back face):
- **Maximize T1 and R2 (left port):**
  - T1 = 0.91736 and R2 = 0.99963. The run takes about 2 s for 256 Adam starts plus about 0.3 s of L-BFGS-B.
  - **T1 equals the ceiling set by the bare back face** (1 − R_bare,s(45°) = 0.917358), so the coating is perfectly
    anti-reflective at 532 nm. Going further needs an AR coating on the back face, p-polarization, or a wedge.
  - Thickness Monte Carlo (1% per layer): T1 mean 0.913, 5th percentile 0.903; R2 stays at 0.9996.
- **Maximize the up port (R1 and T2):** R1 = 0.99984, T2 = 0.91997, again at its back-face ceiling.
- **Both polarizations:** T1 = 0.917 (s) / 0.993 (p), R2 = 0.9997 (s) / 0.992 (p).
- **Engine agreement on the full plate:** stable vs Byrnes to ≤ 6e-15; stable vs tmmax to ≤ 2.5e-6.

**Back-face AR coating** (added after the first round). Set it up with a `[back_coating]` section, as in
`config/example_back_ar.toml`. The design vector then holds the 15 front thicknesses followed by the 4 back
thicknesses, and the two faces are optimized together. Gradients and the reference engine are checked in tests.

| Design (532/1064 nm, s-pol targets) | T1 (s) | R2 (s) | T1 under 1% thickness errors: mean / 5th pct |
|---|---|---|---|
| bare back face (`example.toml`) | 0.91736 (at its ceiling) | 0.99963 | 0.913 / 0.903 |
| + 4-layer back AR (`example_back_ar.toml`) | **0.999998** | 0.99973 | 0.983 / 0.947 |
| + back AR, s and p targets (`example_back_ar_sp.toml`) | 0.9979 (p: 0.9972) | 0.99968 (p: 0.9924) | 0.971 / 0.915 |

The run takes about 3 s (256 Adam starts plus 0.5 s of L-BFGS-B). Things to keep in mind:
- **The back face does two jobs.** It is a perfect AR at 532 nm (R = 0.0000), but it reflects **75% at
  1064 nm**.
  - The merit counts total power into the left port, so R2 now arrives as two parallel beams about 0.8 mm
    apart: 75% reflected at the back face, and 25% from the front coating after a double pass through 1 mm of
    glass.
  - With a bare back face that ghost carried only 8%.
  - If the beams must overlap exactly, add a penalty on back-face reflection at the reflected wavelength.
- **The AR design is more sensitive to thickness errors** (T1 5th percentile 0.947, compared with 0.903 for the
  bare design, which starts from a lower value). A robustness-averaged merit (§9, phase 4) would trade some
  nominal performance for tolerance.
- **Coherent fringe envelope:** T1 has no fringes (both faces are AR at 532 nm). R2 ranges from 0.9962 to
  0.99998 because the substrate now sits between two 1064 nm reflectors.

## Sources
- tmmax repository: https://github.com/bahremsd/tmmax
- PyPI: https://pypi.org/project/tmmax/
- Documentation: https://tmmax.readthedocs.io/
- Paper: arXiv:2507.11341 and JOSS 10.21105/joss.09088
- Byrnes, "Multilayer optical calculations", arXiv:1603.02720 (conventions and the opacity cap)
- Katsidis & Siapkas, Appl. Opt. 41(19):3978 (2002) (incoherent layers)
- JAX 0.7.1 release notes (first Python 3.14 wheels)
