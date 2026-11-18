from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from ifopt.engines import ReferenceEngine, TmmaxEngine
from ifopt.objectives import make_merit, quarter_wave_seed
from ifopt.plate import PlateModel, evaluate_with_engine, make_points, plate_point, plate_point_coherent


def test_coherent_phase_average_equals_incoherent(cfg):
    d = quarter_wave_seed(cfg) * 1.07
    pt = jax.tree.map(lambda a: a[0], make_points(cfg, 2))
    psi = jnp.linspace(0, jnp.pi, 512, endpoint=False)
    d_back = jnp.zeros(0)
    coh = jax.vmap(lambda p: plate_point_coherent(jnp.asarray(d), d_back, pt, p))(psi).mean(axis=0)
    assert np.allclose(coh, plate_point(jnp.asarray(d), d_back, pt), atol=1e-12)


def test_engines_agree_on_full_plate(cfg):
    d = quarter_wave_seed(cfg)
    stable = PlateModel(cfg).coefficients(d)
    for eng, tol in ((ReferenceEngine(), 1e-12), (TmmaxEngine(), 5e-5)):
        other = evaluate_with_engine(cfg, d, eng)
        for c in stable:
            for k in ("s", "p", "mixed"):
                assert abs(stable[c][k] - other[c][k]) < tol, (eng.name, c, k)


def test_opaque_substrate_is_graceful(cfg):
    cfg2 = replace(cfg, plate=replace(cfg.plate, substrate="Ge"))     # Ge: k = 2.58 at 532 nm
    m = PlateModel(cfg2).coefficients(quarter_wave_seed(cfg))
    assert np.isfinite([v["mixed"] for v in m.values()]).all()
    assert m["T1"]["mixed"] < 1e-30 and m["T2"]["mixed"] < 1e-30


def test_transmission_respects_fixed_face_ceiling(cfg):
    model = PlateModel(cfg)
    ceil = model.back_face_ceiling(quarter_wave_seed(cfg))
    rng = np.random.default_rng(0)
    lo, hi = map(np.asarray, cfg.coating.bounds())
    for _ in range(20):
        m = np.asarray(model.mixed(rng.uniform(lo, hi)))
        assert m[0, 1] <= ceil[0] + 1e-12 and m[1, 1] <= ceil[1] + 1e-12


def test_merit_gradient_matches_finite_differences(cfg):
    model = PlateModel(cfg)
    merit, _ = make_merit(cfg, model)
    d = jnp.asarray(quarter_wave_seed(cfg) * 1.05)
    check_grads(merit, (d,), order=1, modes=["rev"], atol=1e-6, rtol=1e-5, eps=1e-3)


def test_short_optimisation_improves_and_respects_bounds(cfg):
    from ifopt.optimize import optimize
    cfg2 = replace(cfg, optimizer=replace(cfg.optimizer, n_starts=16, adam_steps=150, top_k=2))
    model = PlateModel(cfg2)
    merit, _ = make_merit(cfg2, model)
    res = optimize(cfg2, log=lambda *_: None)
    lo, hi = map(np.asarray, cfg2.coating.bounds())
    assert np.all(res.d >= lo - 1e-9) and np.all(res.d <= hi + 1e-9)
    assert res.merit < float(merit(jnp.asarray(quarter_wave_seed(cfg2)))) * 0.1
    ceil = model.back_face_ceiling(res.d)
    assert res.selected["T1"] == pytest.approx(ceil[0], abs=2e-3)   # bare back face is the limit
    assert res.selected["R2"] > 0.999


@pytest.fixture(scope="module")
def cfg_ar(root):
    from ifopt.config import load_config
    return load_config(root / "config" / "example_back_ar.toml")


def test_back_coating_design_vector(cfg_ar):
    assert cfg_ar.back_optimized and len(cfg_ar.design_materials()) == 15 + 4
    x = quarter_wave_seed(cfg_ar)
    front, back = cfg_ar.split(x)
    assert front.shape == (15,) and back.shape == (4,)
    lo, hi = cfg_ar.design_bounds()
    assert len(lo) == len(hi) == 19


def test_back_coating_gradient_and_engines(cfg_ar):
    model = PlateModel(cfg_ar)
    merit, _ = make_merit(cfg_ar, model)
    x = jnp.asarray(quarter_wave_seed(cfg_ar) * 1.03)
    check_grads(merit, (x,), order=1, modes=["rev"], atol=1e-6, rtol=1e-5, eps=1e-3)
    stable = model.coefficients(x)
    ref = evaluate_with_engine(cfg_ar, x, ReferenceEngine())
    assert all(abs(stable[c]["mixed"] - ref[c]["mixed"]) < 1e-12 for c in stable)


def test_back_ar_lifts_bare_face_ceiling(cfg, cfg_ar):
    from ifopt.optimize import optimize
    bare_ceiling = PlateModel(cfg).back_face_ceiling(quarter_wave_seed(cfg))[0]
    c = replace(cfg_ar, optimizer=replace(cfg_ar.optimizer, n_starts=32, adam_steps=300, top_k=3))
    res = optimize(c, log=lambda *_: None)
    assert res.selected["T1"] > 0.995 > bare_ceiling + 0.07
    assert res.selected["R2"] > 0.999
    lo, hi = map(np.asarray, c.design_bounds())
    assert np.all(res.d >= lo - 1e-9) and np.all(res.d <= hi + 1e-9)
