"""TOML round trip, custom materials, CLI and script shims."""
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import ifopt as fo
from ifopt.cli import main
from ifopt.io import dumps_toml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "config").glob("*.toml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_toml_roundtrip(tmp_path, path):
    s = fo.FilterSystem.from_toml(path)
    out = tmp_path / path.name
    s.to_toml(out)
    s2 = fo.FilterSystem.from_toml(out)
    assert s2 == s and s2.to_dict() == s.to_dict()


def test_writer_matches_tomllib():
    data = {"a": 1, "b": 2.5, "c": "x\"y", "d": [1.0, float("inf")], "t": {"k": True, "n": {"z": [1, 2]}},
            "arr": [{"name": "p", "v": [0.1]}, {"name": "q", "v": []}], "odd key": "ok"}
    assert tomllib.loads(dumps_toml(data)) == data


def test_custom_materials_roundtrip(tmp_path):
    csv = tmp_path / "Foo.csv"
    csv.write_text("400,2.0,0.1\n800,1.8,0.2\n")
    lib = fo.MaterialLibrary()
    lib.add_constant("C", 1.7, 0.001)
    lib.add_sellmeier("S", [1.0], [0.01], 300, 2000)
    lib.add_table("Tab", [400, 900], [1.6, 1.5])
    lib.add_csv(csv)
    s = fo.FilterSystem(fo.Substrate("FusedSilica", mode="semi_infinite"), fo.Coating(["C", "S", "Tab", "Foo"]),
                        [fo.Target("t", "T", 1, wavelength_nm=600)], materials=lib)
    sub = tmp_path / "sub"
    sub.mkdir()
    text = s.to_toml(sub / "s.toml")
    assert 'path = "../Foo.csv"' in text
    s2 = fo.FilterSystem.from_toml(sub / "s.toml")
    for m in ("C", "S", "Tab", "Foo"):
        assert s2.library.nk(m, 600)[0] == pytest.approx(s.library.nk(m, 600)[0])


def test_cli_evaluate_and_logging_options(capsys, tmp_path):
    args = ["evaluate", "--config", str(ROOT / "config" / "combiner_45.toml"),
            "--design", str(ROOT / "results" / "combiner_45" / "design.json"), "--engines", ""]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "T1" in out and "0.917358" in out and "left port" in out
    assert main(["-q", *args]) == 0
    assert capsys.readouterr().out == ""
    log = tmp_path / "run.log"
    assert main([*args, "--log-file", str(log)]) == 0
    assert "INFO ifopt.cli" in log.read_text()
    assert main(["evaluate", "--config", str(tmp_path / "missing.toml")]) == 2


def test_cli_optimize_writes_outputs(tmp_path, capsys):
    out = tmp_path / "run"
    assert main(["optimize", "--config", str(ROOT / "config" / "combiner_45.toml"), "--starts", "8", "--steps", "50",
                 "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "[A] Adam" in text and "Thickness Monte Carlo" in text
    for f in ("design.json", "report.txt", "spectra.png", "stack.png", "convergence.png"):
        assert (out / f).exists()
    s = fo.FilterSystem.from_toml(ROOT / "config" / "combiner_45.toml")
    fo.Design.load(out / "design.json", s)


def test_script_shim_runs():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "compute_rt.py"), "--config",
                        str(ROOT / "config" / "combiner_45.toml"), "--engines", ""],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "quarter-wave seed" in r.stdout
