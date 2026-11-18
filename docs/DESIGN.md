# Design decisions

Each decision below comes with its reason and the numbers measured to support it. The evidence is in
`results/stability/stability_report.md`, in the reference runs under `results/`, and in the test suite.

## 1. Optical system (`system.xml`)

- **Geometry:** a plate at 45°, with the two sources hitting **opposite faces**.
  - Source 1 hits the upper-right face. This is the *front* face, which carries the coating.
  - Source 2 hits the lower-left face, the *back* face.
- **Consequence:** source 2 crosses the layers in reverse order, so R and T are needed from both sides of every stack.
  - With absorption, R_front ≠ R_back.
  - T_front = T_back always holds (reciprocity), and the tests use this as a check.
- **Output ports:** left = {T@1, R@2}, up = {R@1, T@2}. The plate works as a dichroic beam combiner.
- **Plane of incidence:** both beams lie in the drawing plane, so s means E perpendicular to the drawing.
- **Automatic check:** `geometry.verify_against_drawio` re-derives this mapping from the XML (arrow colours,
  directions and nearest labels). The combiner configs are tested against it.

## 2. Engines

| Engine | Role | Reason |
|---|---|---|
| **S-matrix** (`engines/stable_engine.py`, complex128) | default for evaluation and optimization | The Redheffer star product uses only the decaying factor e^{iδ} (\|·\| ≤ 1), so it cannot overflow for any loss or thickness. It gives r, t, r′ and t′ from both sides in one pass, and it is differentiable. |
| **tmmax** (pinned to commit `dc4b904`) | cross-check only | See the measured problems below. |
| **Byrnes `tmm`** (float64) | independent reference | Used in the tests and the reports. |

Why tmmax is only a cross-check (all measured):
- **Single precision only.** It hard-codes complex64 and crashes when `jax_enable_x64` is on, so the adapter runs it
  inside `jax.enable_x64(False)`.
- **It fails with absorbing layers.** Its T-matrix returns NaN once Σ Im δ ≈ 89 (2.9 µm of Ge at 532 nm). Its gradients
  are wrong from Σ Im δ ≈ 46, and float32 limits gradient accuracy to about 1e-8/nm.
- **Its incoherent path applies the wrong layer's absorption.** It gives T = 0.44 where the correct value is 0.0011.
- **Other failures:** `jit(grad(tmm()))` fails, a bare interface crashes, and PyPI 1.1.4 multiplies the layer matrices in
  the wrong order.
- **Switching to float64 would not fix the overflow.** A complex128 T-matrix still fails at Σ Im δ ≈ 760.

## 3. Substrate model

- **`incoherent`** (thick plate): R = R_a + T_a·T_a′·R_b·τ² / (1 − R_a′·R_b·τ²) and T = T_a·T_b·τ / (1 − R_a′·R_b·τ²).
  - Here τ = exp(−4π·D·Im q/λ). τ only ever multiplies, so an opaque substrate tends smoothly to T → 0 and R → R_a.
  - The formula is applied per polarization, and the polarizations are mixed afterwards. That mixing is exact for
    isotropic stacks.
- **Fringe envelope:** in the plate modes, the coherent plate is scanned over the substrate phase. Its mean equals the
  incoherent result exactly (Parseval), and its min/max show the etalon risk for single-frequency lasers.
- **`coherent`:** treats the substrate as an etalon at its physical phase. It is meant for analysis, not optimization.
- **`semi_infinite`** turns the substrate "off": the back surface is ignored and T is the power entering the glass.
  **`none`** is a free-standing stack.
- **Back-side incidence:** kx is taken in the incidence medium (the substrate, in the stack modes).
- **Back-face ceiling** (incoherent mode): T ≤ T_back·τ for **any** passive front coating. The report prints it.
  - Example: a bare fused-silica back face at 45° in s-polarization limits T to 0.917358.

## 4. Materials

- **Lookup order:** custom materials in the config → `materials/*.csv` → built-in Sellmeier glasses (FusedSilica,
  N-BK7) → the tmmax database, read-only.
- **No silent clamping:** a lookup outside a table's range raises an error. tmmax instead silently uses the value at
  the end of the table.
- **tmmax database audit findings:**
  - `SiO2` has k = 2e-3 at 532 nm, which would make a 1 mm plate opaque, so the examples use the built-in
    `FusedSilica` Sellmeier instead;
  - `Si` covers only 2–20 µm, and the `Kischkat_*` tables only 1.5–14 µm;
  - `CdS` has negative k;
  - `ZnS`, `Ta2O5`, `SiO`, `CeF3` and `ITO` have k > 1e-3.

## 5. Target table and merit (thin-film design convention)

- **Rows:** each `[[targets]]` row specifies (wavelength or range, AOI or range, polarization, side) → quantity
  R|T|A, kind eq|min|max, value, tolerance and weight.
  - A row expands over its wavelength × AOI grid. `polarization = "both"` adds an s copy and a p copy.
  - The row's weight is split evenly over its expanded points.
- **Shared evaluation points:** targets with the same (λ, AOI, side) share one evaluation point, because one solve
  returns R, T and A for both polarizations.
- **Beams** are named operating conditions. Rows on the same beam share their condition, and reports and
  `report.sums` (e.g. port totals, `T@1 + R@2`) refer to beams.
- **Deviation** d, divided by the tolerance:
  - eq: v − value;
  - min: max(value − v, 0);
  - max: max(v − value, 0).
- **Merit forms:**
  - `power`: Σ w|d|^p / Σ w, reported as MF = (·)^{1/p} (the OptiLayer convention);
  - `minimax`: τ·logΣ w·e^{d²/τ} / Σ w;
  - `log`: −Σ w·log(1 − |d|) / Σ w.
- **One-sided targets stop improving once met,** because their gradient becomes zero, so they build no margin. To
  get margin, set the value above the spec or use `eq`.
  - Example: in the bands design, the pass band still meets its spec in only 67% of samples with 1% thickness errors.

## 6. Optimizer

- **Design vector:** the front thicknesses, followed by the back thicknesses when `[back] optimize = true`. Each layer
  has box bounds per material.
- **Starting designs:**
  - the front starts as a quarter-wave stack at the highest-weight "reflect" row. Its layers are half-wave at λ/2, so a
    532/1064 nm combiner is nearly transparent at 532 nm from the start;
  - the back starts as quarter-wave layers at the "transmit" row (an AR starting point);
  - 25% of the starts are jittered copies of the seed, and the rest are uniform random.
- **Stage A:** Adam over all starts at once (vmapped, with a sigmoid mapping to stay inside the bounds).
  - 256 starts × 600 steps take 2–3 s with 2 target points, and 46 s with 52.
- **Stage B:** L-BFGS-B (scipy, native bounds) polishes the best starts using exact float64 gradients. Its line search
  needs float64.
- **Optional:** differential evolution evaluating whole populations of designs at once.

## 7. Validation

- **Analytic checks:** Fresnel at 45° (s and p), zero p-reflectance at Brewster's angle, closed-form quarter-wave stack.
- **Invariants:** R + T = 1 for lossless stacks, reciprocity, R_front ≠ R_back with loss, s = p at 0° in every
  substrate mode.
- **Cross-engine:**
  - stable vs Byrnes: ≤ 1e-10 on random stacks including metal-like layers, in both directions, for the full plate and
    the stack modes;
  - stable vs tmmax: ≤ 5e-5.
- **Other checks:** coherent phase average = incoherent result; `check_grads` on the merit; regression of the reference
  designs; stress sweeps of thick absorbers up to 60 µm.

## 8. Reference results (`results/<config>/`)

| Config | Result | Note |
|---|---|---|
| `combiner_45` | T1 = 0.917358, R2 = 0.999629 (s) | T1 is at the bare back-face ceiling |
| `combiner_45_up_port` | R1 = 0.999839, T2 = 0.919968 | T2 is at its ceiling |
| `combiner_45_back_ar` | T1 = 0.999998, R2 = 0.999730 | the back face also reflects 75% at 1064 nm, so R2 leaves as two beams about 0.8 mm apart; T1 has a 5th percentile of 0.947 under 1% thickness errors |
| `combiner_45_back_ar_sp` | T1 = 0.9979 / 0.9972, R2 = 0.9997 / 0.9924 (s / p) | targets hold for both polarizations |
| `normal_dichroic_bands` | pass band T = 0.9892–0.9999, reflect band R ≥ 0.99975 at 0° | all requirements met; semi-infinite substrate |
