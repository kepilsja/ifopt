import jax.numpy as jnp
import numpy as np
import pytest

from ifopt.engines import ReferenceEngine, StableEngine, TmmaxEngine
from ifopt.engines.stable_engine import stack_powers

KX45 = np.sin(np.pi / 4)


def f(n_in, layers, n_out, d, kx, lam, pol):
    return [float(x) for x in stack_powers(n_in, jnp.asarray(layers, complex), n_out, jnp.asarray(d, float), kx, lam, pol)]


@pytest.mark.parametrize("pol", [0, 1])
def test_single_interface_fresnel(pol):
    n1, n2, th = 1.0, 1.5, np.pi / 4
    th2 = np.arcsin(n1 * np.sin(th) / n2)
    if pol == 0:
        r = (n1 * np.cos(th) - n2 * np.cos(th2)) / (n1 * np.cos(th) + n2 * np.cos(th2))
    else:
        r = (n2 * np.cos(th) - n1 * np.cos(th2)) / (n2 * np.cos(th) + n1 * np.cos(th2))
    R, T, Rb, Tb = f(n1, [], n2, [], n1 * np.sin(th), 500.0, pol)
    assert R == pytest.approx(r ** 2, abs=1e-14) and R + T == pytest.approx(1, abs=1e-14)
    assert Rb == pytest.approx(R, abs=1e-14) and Tb == pytest.approx(T, abs=1e-14)


def test_brewster_angle_p_reflectance_vanishes():
    n = 1.5
    R = f(1.0, [], n, [], np.sin(np.arctan(n)), 600.0, 1)[0]
    assert R < 1e-28


def test_quarter_wave_stack_closed_form():
    nH, nL, ns, N, lam = 2.35, 1.45, 1.52, 6, 1000.0
    layers = [nH, nL] * N + [nH]
    d = [lam / 4 / n for n in layers]
    Y = (nH / nL) ** (2 * N) * nH ** 2 / ns
    R = f(1.0, layers, ns, d, 0.0, lam, 0)[0]
    assert R == pytest.approx(((1 - Y) / (1 + Y)) ** 2, rel=1e-12)


def test_normal_incidence_s_equals_p():
    layers, d = [2.3 + 0.01j, 1.46, 0.05 + 3.4j], [80.0, 120.0, 20.0]
    assert np.allclose(f(1.0, layers, 1.52, d, 0.0, 600.0, 0), f(1.0, layers, 1.52, d, 0.0, 600.0, 1), atol=1e-14)


def random_stack(rng, lossy):
    L = rng.integers(1, 25)
    n = rng.uniform(1.3, 2.6, L) + 1j * (rng.uniform(0, 0.05, L) if lossy else 0)
    if lossy:
        n[rng.integers(0, L)] = 0.06 + 3.4j          # one metal-like layer
    return n, rng.uniform(5, 300, L)


@pytest.mark.parametrize("lossy", [False, True])
@pytest.mark.parametrize("pol", [0, 1])
def test_stable_matches_byrnes_both_directions(lossy, pol):
    rng = np.random.default_rng(10 + 2 * lossy + pol)
    ref = ReferenceEngine()
    for _ in range(15):
        n, d = random_stack(rng, lossy)
        lam = rng.uniform(400, 1200)
        a = f(1.0, n, 1.46, d, KX45, lam, pol)
        b = ref.face(1.0, n, 1.46, d, KX45, lam, pol)
        assert np.allclose(a, b, rtol=1e-10, atol=1e-12)


def test_tmmax_matches_within_single_precision():
    rng = np.random.default_rng(3)
    tm = TmmaxEngine()
    for pol in (0, 1):
        for _ in range(5):
            n, d = random_stack(rng, lossy=True)
            a = f(1.0, n, 1.46, d, KX45, 700.0, pol)
            assert np.allclose(a, tm.face(1.0, n, 1.46, d, KX45, 700.0, pol), atol=3e-5)


def test_invariants():
    rng = np.random.default_rng(5)
    for pol in (0, 1):
        n, d = random_stack(rng, lossy=False)
        R, T, Rb, Tb = f(1.0, n, 1.46, d, KX45, 800.0, pol)
        assert R + T == pytest.approx(1, abs=1e-12) and R == pytest.approx(Rb, abs=1e-12)
        n, d = random_stack(rng, lossy=True)
        R, T, Rb, Tb = f(1.0, n, 1.46, d, KX45, 800.0, pol)
        assert T == pytest.approx(Tb, rel=1e-10)              # reciprocity, also with loss
        assert R + T <= 1 + 1e-12 and Rb + Tb <= 1 + 1e-12
        assert abs(R - Rb) > 1e-4                             # absorption breaks R symmetry


def test_thick_absorber_is_finite_and_opaque():
    for d in [1e3, 1e4, 1e5, 1e6]:
        R, T, Rb, Tb = f(1.0, [3.46 + 2.58j], 1.46, [d], KX45, 532.0, 0)
        assert np.isfinite([R, T, Rb, Tb]).all() and T < 1e-25
        assert R == pytest.approx(f(1.0, [], 3.46 + 2.58j, [], KX45, 532.0, 0)[0], abs=1e-12)
