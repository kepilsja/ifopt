"""Public scripting API: spec objects, Design, evaluate/optimize/spectrum, logging and x64 policy."""
import ast
import json
import logging
from dataclasses import FrozenInstanceError
from pathlib import Path

import jax
import numpy as np
import pytest

import ifopt as fo
from ifopt.engines.stable_engine import stack_powers

ROOT = Path(__file__).resolve().parents[1]


def combiner_in_code(lib=None):
    b1 = fo.Beam("1", wavelength_nm=532, aoi_deg=45, side="front", polarization="s")
    b2 = fo.Beam("2", wavelength_nm=1064, aoi_deg=45, side="back", polarization="s")
    return fo.FilterSystem(
        substrate=fo.Substrate("FusedSilica", mode="incoherent", thickness_mm=1),
        front=fo.Coating(fo.alternating("TiO2", "FusedSilica", 15), bounds_nm={"default": (5, 400)}),
        targets=[b1.eq("T", 1.0, name="T1", weight=1), b2.eq("R", 1.0, name="R2")],
        sums=[fo.ReportSum("left port", ["T@1", "R@2"]), ("up port", ["R@1", "T@2"])],
        materials=lib)


@pytest.fixture(scope="module")
def toml_system():
    return fo.FilterSystem.from_toml(ROOT / "config" / "combiner_45.toml")


def test_code_built_system_equals_toml(toml_system):
    s = combiner_in_code()
    assert s == toml_system
    assert s.to_dict() == toml_system.to_dict()
    d = fo.Design.quarter_wave(s)
    a, b = fo.evaluate(s, d), fo.evaluate(toml_system, d)
    assert a.merit == b.merit and np.array_equal(a.response, b.response)


def test_beam_helpers_and_inheritance():
    b = fo.Beam("pass", wavelength_nm=(520, 545), points=6)
    t = b.min("T", 0.98)
    assert t.name == "T@pass" and t.kind == "min" and t.beam == "pass" and t.points == 6
    s = fo.FilterSystem(fo.Substrate("FusedSilica", mode="semi_infinite"), fo.Coating(["TiO2", "FusedSilica"]),
                        [t, fo.Target("absorb", "A", 0.0, kind="max", beam="pass")])
    assert s.targets[1].wavelength_nm == (520.0, 545.0) and s.targets[1].points == 6
    assert s.table.n_points == 6


def test_validation_at_construction():
    with pytest.raises(ValueError, match="quantity"):
        fo.Target("x", "Q", 1, wavelength_nm=500)
    with pytest.raises(ValueError, match="mode"):
        fo.Substrate("FusedSilica", mode="thick")
    with pytest.raises(ValueError, match="optimize"):
        fo.Coating(["TiO2"], optimize=False)
    with pytest.raises(ValueError, match="finite substrate"):
        fo.FilterSystem(fo.Substrate("FusedSilica", mode="none"), fo.Coating(["TiO2"]),
                        [fo.Target("t", "T", 1, wavelength_nm=500)], back=fo.Coating(["TiO2"]))
    with pytest.raises(fo.MaterialRangeError):          # caught at construction, not inside JAX
        fo.FilterSystem(fo.Substrate("FusedSilica", mode="semi_infinite"), fo.Coating(["Si"]),
                        [fo.Target("t", "T", 1, wavelength_nm=532)])
    s = combiner_in_code()
    with pytest.raises(FrozenInstanceError):
        s.ambient = "Vacuum"
    with pytest.raises(ValueError, match="duplicate"):
        s.add_targets(fo.Target("T1", "R", 0, wavelength_nm=600))


def test_material_objects_are_registered():
    ag = fo.ConstantMaterial(name="AgLike", source="test", n=0.05, k=3.4)
    s = fo.FilterSystem(fo.Substrate("FusedSilica", mode="semi_infinite"), fo.Coating(["TiO2", ag]),
                        [fo.Target("t", "R", 1, wavelength_nm=600)])
    assert s.front.materials == ("TiO2", "AgLike") and s.library.nk("AgLike", 600)[0] == pytest.approx(0.05 + 3.4j)
    with pytest.raises(ValueError, match="already registered"):
        s.library.add(fo.ConstantMaterial(name="AgLike", source="other", n=0.1, k=3.0))
    s.library.add_table("Dye", [400, 800], [1.5, 1.5], [0.01, 0.0])
    assert s.library.nk("Dye", 600)[0] == pytest.approx(1.5 + 0.005j)


def test_replace_creates_validated_variant(toml_system):
    v = toml_system.replace(substrate=fo.Substrate("FusedSilica", mode="semi_infinite"))
    assert v.mode == "semi_infinite" and toml_system.mode == "incoherent"
    assert v.library is toml_system.library


def test_design_roundtrips(tmp_path, toml_system):
    d = fo.Design.quarter_wave(toml_system)
    assert fo.Design.from_x(toml_system, d.x(toml_system)) == d
    d.save(tmp_path / "d.json", toml_system, merit=1.0)
    assert fo.Design.load(tmp_path / "d.json", toml_system) == d
    old = fo.Design.load(ROOT / "results" / "combiner_45" / "design.json", toml_system)
    assert len(old.front_nm) == 15
    with pytest.raises(ValueError, match="front layers"):
        fo.Design([1.0, 2.0]).x(toml_system)


def test_evaluation_contents(toml_system):
    d = fo.Design.load(ROOT / "results" / "combiner_45" / "design.json", toml_system)
    ev = fo.evaluate(toml_system, d, engines=("reference",), label="saved")
    t1 = next(t for t in ev.targets if t.name == "T1")
    assert t1.achieved_min == pytest.approx(0.917358, abs=1e-6)
    assert ev.beams["1"].T[0] == pytest.approx(t1.achieved_min)
    assert ev.sums["left port"] == pytest.approx(1.916987, abs=1e-6)
    assert ev.ceiling["1"] == pytest.approx(0.917358, abs=1e-6)
    assert np.nanmax(ev.cross_check["reference"]) < 1e-12
    assert "T1" in ev.report() and "left port" in ev.report()
    df = ev.to_pandas()
    assert list(df["target"]) == ["T1", "R2"] and df["achieved"].iloc[0] == pytest.approx(t1.achieved_min)
    json.dumps(ev.to_dict())


def test_spectrum_matches_target_points(toml_system):
    d = fo.Design.quarter_wave(toml_system)
    ev = fo.evaluate(toml_system, d, extras=False)
    sp = fo.spectrum(toml_system, d, wavelengths_nm=[532.0], aoi_deg=45, side="front", polarization="s")
    assert sp.T[0] == pytest.approx(ev.beams["1"].T[0], abs=1e-14)
    sweep = fo.spectrum(toml_system, d, wavelengths_nm=np.linspace(500, 560, 7), aoi_deg=45)
    assert sweep.mixed.shape == (7, 3) and np.allclose(sweep.R + sweep.T + sweep.A, 1)
    assert fo.beam_spectrum(toml_system, d, "1", n=11).mixed.shape == (11, 3)
    assert fo.angle_scan(toml_system, d, "2", n=5).aoi_deg.shape == (5,)


def test_stack_rt_matches_engine_and_fresnel():
    r = fo.stack_rt([], [], wavelength_nm=500, n_out=1.5, aoi_deg=0.0)
    assert r.R == pytest.approx(0.04) and r.T == pytest.approx(0.96)
    n = [2.3 + 0.01j, 1.46]
    d = [80.0, 120.0]
    r = fo.stack_rt(n, d, wavelength_nm=600, n_out=1.52, aoi_deg=30, polarization="p")
    F = stack_powers(1.0, jax.numpy.asarray(n), 1.52, jax.numpy.asarray(d), np.sin(np.radians(30)), 600.0, 1)
    assert (r.R, r.T, r.Rb, r.Tb) == pytest.approx(tuple(float(v) for v in F))


def test_optimize_api_logs_and_accepts_start(caplog, toml_system):
    seed = fo.Design.quarter_wave(toml_system)
    with caplog.at_level(logging.INFO, logger="ifopt"):
        opt = fo.optimize(toml_system, start=seed, settings=fo.OptimizerSettings(n_starts=8, adam_steps=100, top_k=2))
    msgs = [r.getMessage() for r in caplog.records if r.name.startswith("ifopt")]
    assert any(m.startswith("[A] Adam") for m in msgs) and any(m.startswith("[B] L-BFGS-B") for m in msgs)
    assert isinstance(opt.design, fo.Design) and opt.evaluation.merit == pytest.approx(opt.merit)
    assert opt.mf <= fo.evaluate(toml_system, seed).mf


def test_x64_is_forced_inside_api(toml_system):
    d = fo.Design.quarter_wave(toml_system)
    with jax.enable_x64(False):
        ev = fo.evaluate(toml_system, d, extras=False)
    assert ev.response.dtype == np.float64


def test_library_never_prints():
    calls = []
    for path in (ROOT / "src" / "ifopt").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                calls.append(f"{path.name}:{node.lineno}")
    assert not calls, f"print() calls in the library: {calls}"


def test_enable_logging_attaches_one_handler():
    import io
    buf = io.StringIO()
    fo.enable_logging("INFO", stream=buf)
    fo.enable_logging("INFO", stream=buf)
    log = logging.getLogger("ifopt")
    try:
        assert sum(getattr(h, "_ifopt_enable_logging", False) for h in log.handlers) == 1
        logging.getLogger("ifopt.test").info("hello")
        assert "hello" in buf.getvalue()
    finally:
        for h in [h for h in log.handlers if getattr(h, "_ifopt_enable_logging", False)]:
            log.removeHandler(h)
        log.setLevel(logging.NOTSET)


def test_jittered_starts_center_on_user_start(toml_system):
    from ifopt.optimizer import initial_designs
    rng = np.random.default_rng(0)
    seed = fo.Design.quarter_wave(toml_system).x(toml_system) * 1.3
    D = initial_designs(toml_system, 400, rng, extra=[seed])
    n_fixed = 2                                         # QW seed + user start (no configured thicknesses)
    jit = D[n_fixed:n_fixed + 100]
    assert np.allclose(D[1], np.clip(seed, *map(np.asarray, toml_system.design_bounds())))
    # centred on the user start (ratio ~1), not on the quarter-wave seed (ratio ~1.3 away)
    assert abs(np.median(jit / D[1]) - 1.0) < 0.03
    assert abs(np.median(jit / D[0]) - 1.0) > 0.2


def test_write_run_layout(tmp_path, toml_system):
    opt = fo.optimize(toml_system, settings=fo.OptimizerSettings(n_starts=8, adam_steps=50, top_k=2))
    paths = fo.write_run(tmp_path / "run", toml_system, opt, extra={"note": "x"}, mc_samples=50)
    for f in ("config.toml", "design.json", "report.txt", "spectra.png", "stack.png", "convergence.png"):
        assert (tmp_path / "run" / f).exists()
    assert fo.FilterSystem.from_toml(paths["config"]) == toml_system
    assert json.loads(paths["design"].read_text())["note"] == "x"
