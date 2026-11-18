import numpy as np
import pytest

from ifopt.geometry import OUTPUT_PORT, PORTS, verify_against_drawio
from ifopt.materials import MaterialLibrary, MaterialRangeError


def test_system_xml_matches_hardcoded_mapping(root):
    assert verify_against_drawio(root / "system.xml") == OUTPUT_PORT
    assert set(PORTS["left"]) == {k for k, v in OUTPUT_PORT.items() if v == "left"}


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
