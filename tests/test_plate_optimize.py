from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from ifopt.engines import ReferenceEngine, TmmaxEngine
from ifopt.objectives import make_merit, quarter_wave_seed, row_summary
from ifopt.optimizer import run_optimization as optimize
from ifopt.plate import Model, evaluate_with_engine, plate_point, plate_point_coherent


def short(cfg, starts=16, steps=150, top_k=2):
    return replace(cfg, optimizer=replace(cfg.optimizer, n_starts=starts, adam_steps=steps, top_k=top_k))


def test_coherent_phase_average_equals_incoherent(cfg):
    model = Model(cfg)
    x = jnp.asarray(quarter_wave_seed(cfg) * 1.07)
    d_f, d_b = cfg.split(x)
    for i in range(cfg.table.n_points):
        pt = jax.tree.map(lambda a: a[i], model.points)
        psi = jnp.linspace(0, jnp.pi, 512, endpoint=False)
        coh = jax.vmap(lambda p: plate_point_coherent(d_f, d_b, pt, p))(psi).mean(axis=0)
        assert np.allclose(coh, plate_point(d_f, d_b, pt), atol=1e-12)


@pytest.mark.parametrize("name", ["cfg", "cfg_ar"])
def test_engines_agree_on_full_plate(name, request):
    cfg = request.getfixturevalue(name)
    x = quarter_wave_seed(cfg)
    stable = np.asarray(Model(cfg).response(x))
    assert np.abs(evaluate_with_engine(cfg, x, ReferenceEngine()) - stable).max() < 1e-12
    assert np.abs(evaluate_with_engine(cfg, x, TmmaxEngine()) - stable).max() < 5e-5


def test_opaque_substrate_is_graceful(cfg):
    cfg2 = replace(cfg, substrate=replace(cfg.substrate, material="Ge"))     # Ge: k = 2.58 at 532 nm
    res = np.asarray(Model(cfg2).response(quarter_wave_seed(cfg)))
    assert np.isfinite(res).all() and res[:, :, 1].max() < 1e-30


def test_transmission_respects_back_face_ceiling(cfg):
    model = Model(cfg)
    x0 = quarter_wave_seed(cfg)
    ceil = model.back_face_ceiling(x0)                                       # (P, 2)
    rng = np.random.default_rng(0)
    lo, hi = map(np.asarray, cfg.design_bounds())
    for _ in range(20):
        T = np.asarray(model.response(rng.uniform(lo, hi)))[:, :, 1]
        assert np.all(T <= ceil + 1e-12)


@pytest.mark.parametrize("name", ["cfg", "cfg_ar"])
def test_merit_gradient_matches_finite_differences(name, request):
    cfg = request.getfixturevalue(name)
    merit, _ = make_merit(Model(cfg))
    x = jnp.asarray(quarter_wave_seed(cfg) * 1.03)
    check_grads(merit, (x,), order=1, modes=["rev"], atol=1e-6, rtol=1e-5, eps=1e-3)


def test_combiner_regression(cfg):
    """Migrated combiner config reproduces the original result: T1 at the bare back-face ceiling."""
    model = Model(cfg)
    res = optimize(short(cfg))
    rows = {r["name"]: r for r in res.rows}
    ceiling = model.back_face_ceiling(res.d)[cfg.table.beams["1"].points[0], 0]
    assert rows["T1"]["achieved_min"] == pytest.approx(0.917358, abs=2e-3)
    assert rows["T1"]["achieved_min"] == pytest.approx(ceiling, abs=2e-3)
    assert rows["R2"]["achieved_min"] > 0.999
    lo, hi = map(np.asarray, cfg.design_bounds())
    assert np.all(res.d >= lo - 1e-9) and np.all(res.d <= hi + 1e-9)


def test_back_ar_lifts_the_ceiling(cfg_ar):
    assert cfg_ar.back_optimized and len(cfg_ar.design_materials()) == 15 + 4
    res = optimize(short(cfg_ar, starts=32, steps=300, top_k=3))
    rows = {r["name"]: r for r in res.rows}
    assert rows["T1"]["achieved_min"] > 0.995 and rows["R2"]["achieved_min"] > 0.999


def test_row_summary_shares_sum_to_one(cfg):
    rows = row_summary(Model(cfg), quarter_wave_seed(cfg))
    assert sum(r["share"] for r in rows) == pytest.approx(1.0)
