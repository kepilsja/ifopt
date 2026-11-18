"""Substrate modes and the single-source normal-incidence case."""
import textwrap
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ifopt.config import load_config
from ifopt.engines import ReferenceEngine, TmmaxEngine
from ifopt.engines.stable_engine import stack_powers
from ifopt.objectives import quarter_wave_seed
from ifopt.optimizer import run_optimization as optimize
from ifopt.plate import Model, evaluate_with_engine, physical_phase, plate_point_coherent

CFG = """
[system]
ambient = "Air"
[substrate]
material = "FusedSilica"
mode = "{mode}"
thickness_mm = 1.0
[front]
materials = ["TiO2", "FusedSilica", "TiO2", "FusedSilica", "TiO2", "FusedSilica", "TiO2", "FusedSilica", "TiO2"]
[front.bounds_nm]
default = [5.0, 400.0]
[[targets]]
name = "front"
side = "front"
wavelength_nm = [{lo}, {hi}]
points = 3
aoi_deg = {aoi}
polarization = "both"
quantity = "R"
kind = "min"
value = 0.99
[[targets]]
name = "back"
side = "back"
wavelength_nm = 700.0
aoi_deg = {aoi}
polarization = "unpolarized"
quantity = "T"
value = 1.0
[optimizer]
n_starts = 16
adam_steps = 200
top_k = 2
"""


def make(tmp_path, mode, aoi=20.0, lo=1000.0, hi=1100.0):
    p = tmp_path / f"{mode}_{aoi}.toml"
    p.write_text(textwrap.dedent(CFG.format(mode=mode, aoi=aoi, lo=lo, hi=hi)))
    return load_config(p)


@pytest.mark.parametrize("mode", ["semi_infinite", "none"])
def test_stack_modes_match_stack_powers(tmp_path, mode):
    cfg = make(tmp_path, mode)
    x = quarter_wave_seed(cfg) * 1.05
    res = np.asarray(Model(cfg).response(x))
    lib, t = cfg.library, cfg.table
    for i in range(t.n_points):
        lam, aoi, front = t.point_lam[i], t.point_aoi[i], t.point_front[i]
        n_amb, n_exit = lib.nk(cfg.ambient, lam)[0], lib.nk(cfg.exit_material, lam)[0]
        n_inc = n_amb if front else n_exit
        kx = n_inc.real * np.sin(np.radians(aoi))                    # kx in the incidence medium
        n_lay = jnp.asarray([lib.nk(m, lam)[0] for m in cfg.front.materials])
        for pol in (0, 1):
            F = stack_powers(n_amb, n_lay, n_exit, jnp.asarray(x), kx, lam, pol)
            R, T = (F.R, F.T) if front else (F.Rb, F.Tb)
            assert res[i, pol, 0] == pytest.approx(float(R), abs=1e-14)
            assert res[i, pol, 1] == pytest.approx(float(T), abs=1e-14)
    if mode == "none":
        assert cfg.exit_material == "Air"


@pytest.mark.parametrize("mode", ["incoherent", "coherent", "semi_infinite", "none"])
def test_normal_incidence_s_equals_p(tmp_path, mode):
    cfg = make(tmp_path, mode, aoi=0.0)
    res = np.asarray(Model(cfg).response(quarter_wave_seed(cfg) * 0.97))
    assert np.allclose(res[:, 0, :], res[:, 1, :], atol=1e-12)


def test_coherent_mode_uses_physical_phase(tmp_path):
    cfg = make(tmp_path, "coherent")
    model = Model(cfg)
    x = quarter_wave_seed(cfg)
    d_f, d_b = cfg.split(x)
    res = np.asarray(model.response(x))
    for i in range(cfg.table.n_points):
        pt = jax.tree.map(lambda a: a[i], model.points)
        assert np.allclose(res[i], plate_point_coherent(d_f, d_b, pt, physical_phase(pt)), atol=1e-14)


@pytest.mark.parametrize("mode", ["semi_infinite", "none"])
def test_stack_modes_cross_check_with_other_engines(tmp_path, mode):
    cfg = make(tmp_path, mode)
    x = quarter_wave_seed(cfg)
    stable = np.asarray(Model(cfg).response(x))
    assert np.abs(evaluate_with_engine(cfg, x, ReferenceEngine()) - stable).max() < 1e-12
    assert np.abs(evaluate_with_engine(cfg, x, TmmaxEngine()) - stable).max() < 5e-5


def test_single_source_normal_incidence_mirror(tmp_path):
    """One target row: a laser-line HR at 0 deg on a semi-infinite substrate (substrate 'off')."""
    p = tmp_path / "hr.toml"
    p.write_text(textwrap.dedent("""
        [substrate]
        material = "FusedSilica"
        mode = "semi_infinite"
        [front]
        # 17 layers: QW peak R = ((1-Y)/(1+Y))^2 with Y = (nH/nL)^16 nH^2/ns ~ 7800 -> R ~ 0.9995
        materials = ["TiO2", "FusedSilica", "TiO2", "FusedSilica", "TiO2", "FusedSilica", "TiO2", "FusedSilica",
                     "TiO2", "FusedSilica", "TiO2", "FusedSilica", "TiO2", "FusedSilica", "TiO2", "FusedSilica", "TiO2"]
        [[targets]]
        name = "HR"
        wavelength_nm = 1064.0
        aoi_deg = 0.0
        quantity = "R"
        kind = "min"
        value = 0.999
        [optimizer]
        n_starts = 8
        adam_steps = 100
        top_k = 2
        """))
    cfg = load_config(p)
    assert cfg.table.n_points == 1 and not cfg.has_back_face
    seed = quarter_wave_seed(cfg)
    assert float(Model(cfg).target_values(seed)[0]) > 0.999          # the QW stack already meets it
    res = optimize(cfg)
    assert res.rows[0]["achieved_min"] >= 0.999 and res.merit == pytest.approx(0.0, abs=1e-14)


def test_bands_example_short_run(root):
    cfg = load_config(root / "config" / "normal_dichroic_bands.toml")
    assert cfg.mode == "semi_infinite" and cfg.table.n_points == 52
    from ifopt.objectives import make_merit
    merit, _ = make_merit(Model(cfg))
    m0 = float(merit(jnp.asarray(quarter_wave_seed(cfg))))
    res = optimize(replace(cfg, optimizer=replace(cfg.optimizer, n_starts=16, adam_steps=200, top_k=2)))
    assert res.merit < 0.1 * m0
