import numpy as np
import pytest

from ifopt.geometry import OUTPUT_PORT, expected_port_terms, verify_against_drawio
from ifopt.materials import MaterialLibrary, MaterialRangeError


def test_system_xml_matches_hardcoded_mapping(root):
    assert verify_against_drawio(root / "system.xml") == OUTPUT_PORT


@pytest.mark.parametrize("name", ["combiner_45", "combiner_45_up_port", "combiner_45_back_ar", "combiner_45_back_ar_sp"])
def test_combiner_configs_follow_system_xml(root, name):
    from ifopt.config import load_config
    cfg = load_config(root / "config" / f"{name}.toml")
    beams = cfg.table.beams
    assert beams["1"].side == "front" and beams["2"].side == "back"     # source 1 faces the coated face
    assert all(b.aois_deg == (45.0,) for b in beams.values())
    expected = expected_port_terms(root / "system.xml")
    sums = {s.name.split()[0]: set(s.terms) for s in cfg.report_sums}
    assert sums == expected


def test_sellmeier_fused_silica(lib):
    n = lib.nk("FusedSilica", [532.0, 1064.0])
    assert abs(n[0].real - 1.46071) < 2e-5 and abs(n[1].real - 1.44963) < 2e-5
    assert np.all(n.imag == 0)


def test_out_of_range_raises_instead_of_clamping(lib):
    with pytest.raises(MaterialRangeError):
        lib.nk("Si", 532.0)          # tmmax table covers 2-20 um only


def test_csv_and_constant_materials(tmp_path):
    (tmp_path / "Foo.csv").write_text("# wavelength_nm,n,k\n400,2.0,0.1\n800,1.8,0.2\n")
    lib = MaterialLibrary.from_config_entries(
        [{"name": "C", "type": "constant", "n": 1.5, "k": 0.01}], tmp_path, tmp_path)
    assert lib.nk("C", 1234.0)[0] == pytest.approx(1.5 + 0.01j)
    assert lib.nk("Foo", 600.0)[0] == pytest.approx(1.9 + 0.15j)
    with pytest.raises(MaterialRangeError):
        lib.nk("Foo", 900.0)
