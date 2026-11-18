"""Target table parsing / expansion, merit forms and validation errors."""
import textwrap

import jax.numpy as jnp
import numpy as np
import pytest

from ifopt.config import ObjectiveCfg, load_config
from ifopt.objectives import merit_from_deviation
from ifopt.targets import TargetTable, deviation, parse_targets

BASE = """
[substrate]
material = "FusedSilica"
mode = "{mode}"
[front]
materials = ["TiO2", "FusedSilica", "TiO2"]
{extra}
"""


def write_cfg(tmp_path, targets: str, mode="semi_infinite", extra="", name="c.toml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(BASE.format(mode=mode, extra=extra)) + textwrap.dedent(targets))
    return p


def test_expansion_ranges_both_and_weights():
    rows = parse_targets([
        {"name": "band", "wavelength_nm": [500, 600], "points": 11, "aoi_deg": [0, 10], "aoi_points": 3,
         "polarization": "both", "quantity": "T", "kind": "min", "value": 0.9, "weight": 2.0},
        {"name": "line", "wavelength_nm": 550.0, "aoi_deg": 0.0, "quantity": "R", "kind": "max", "value": 0.01}])
    t = TargetTable.build(rows)
    assert rows[0].n_points == 11 * 3 * 2
    assert t.t_point.size == 66 + 1
    assert t.n_points == 33                         # 550 nm / 0 deg shared between both rows
    assert t.t_weight[t.t_row == 0].sum() == pytest.approx(2.0)
    assert t.t_weight[t.t_row == 1].sum() == pytest.approx(1.0)
    assert set(map(tuple, t.t_polw[t.t_row == 0])) == {(1.0, 0.0), (0.0, 1.0)}


def test_beam_inheritance_and_conflict():
    rows = parse_targets([{"name": "a", "beam": "b", "wavelength_nm": 633, "aoi_deg": 10, "polarization": "p",
                           "quantity": "T", "value": 1},
                          {"name": "c", "beam": "b", "quantity": "R", "kind": "max", "value": 0.0}])
    assert rows[1].wavelengths_nm == (633.0,) and rows[1].aois_deg == (10.0,) and rows[1].polarization == "p"
    with pytest.raises(ValueError, match="conflicts"):
        parse_targets([{"name": "a", "beam": "b", "wavelength_nm": 633, "quantity": "T", "value": 1},
                       {"name": "c", "beam": "b", "wavelength_nm": 532, "quantity": "R", "value": 0}])


@pytest.mark.parametrize("row, msg", [
    ({"name": "x", "wavelength_nm": 500, "quantity": "Q", "value": 1}, "quantity"),
    ({"name": "x", "wavelength_nm": 500, "quantity": "T", "kind": "ge", "value": 1}, "kind"),
    ({"name": "x", "wavelength_nm": 500, "quantity": "T", "value": 1, "side": "left"}, "side"),
    ({"name": "x", "wavelength_nm": [600, 500], "points": 3, "quantity": "T", "value": 1}, "range"),
    ({"name": "x", "wavelength_nm": 500, "quantity": "T", "value": 1, "polarization": "circular"}, "polarization"),
    ({"name": "x", "quantity": "T", "value": 1}, "wavelength_nm"),
    ({"name": "x", "wavelength_nm": 500, "quantity": "T"}, "value"),
])
def test_row_validation(row, msg):
    with pytest.raises(ValueError, match=msg):
        parse_targets([row])


def test_deviation_kinds():
    v = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    kind = np.array([0, 0, 1, 1, 2, 2])
    value = np.array([0.7, 0.3, 0.7, 0.3, 0.7, 0.3])
    d = deviation(v, kind, value, np.full(6, 0.1))
    assert np.allclose(d, [-2, 2, 2, 0, 0, 2])


def test_merit_forms():
    d, w = jnp.array([0.0, 0.1, -0.2]), jnp.array([1.0, 1.0, 2.0])
    assert float(merit_from_deviation(d, w, ObjectiveCfg("power", 2.0))) == pytest.approx((0.01 + 2 * 0.04) / 4)
    assert float(merit_from_deviation(d, w, ObjectiveCfg("power", 4.0))) == pytest.approx((1e-4 + 2 * 1.6e-3) / 4)
    mm = float(merit_from_deviation(d, w, ObjectiveCfg("minimax", 2.0, 1e-3)))
    assert 0.03 < mm <= 0.04 + 1e-9                     # close to the worst d^2 = 0.04
    lg = float(merit_from_deviation(jnp.array([-0.1]), jnp.array([1.0]), ObjectiveCfg("log")))
    assert lg == pytest.approx(-np.log(0.9))


def test_config_validation(tmp_path):
    row = """
    [[targets]]
    name = "t"
    wavelength_nm = 550
    quantity = "T"
    value = 1
    """
    with pytest.raises(ValueError, match="finite substrate"):
        load_config(write_cfg(tmp_path, row, mode="semi_infinite", extra='[back]\nmaterials = ["TiO2"]'))
    with pytest.raises(ValueError, match="single-point"):
        load_config(write_cfg(tmp_path, row.replace("wavelength_nm = 550", "wavelength_nm = [500, 600]\npoints = 3")
                              + '[[report.sums]]\nname = "s"\nterms = ["T@t"]\n'))
    old = tmp_path / "old.toml"
    old.write_text('[source1]\nwavelength_nm = 532\n')
    with pytest.raises(ValueError, match="old config format"):
        load_config(old)
