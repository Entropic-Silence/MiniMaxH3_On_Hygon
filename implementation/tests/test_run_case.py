"""Keep the wrapper's provenance, timing and stdout in the requested run directory."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def test_run_logs_survive_binary_provenance_scan(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/run_case.py"
    spec = importlib.util.spec_from_file_location("h3_run_case_test", path)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    entry = tmp_path / "InfiniLM/examples/minimax_h3.py"
    entry.parent.mkdir(parents=True)
    entry.write_text('print("compute.h3_transformer (49 calls, diagnostic)")\n')
    (tmp_path / "versions.json").write_text('{"test": true}')
    binaries = tmp_path / ".infini/lib"
    binaries.mkdir(parents=True)
    (binaries / "fixture.so").write_bytes(b"test provenance only")
    run_dir = tmp_path / "logs/run"
    monkeypatch.setenv("H3_LOG_DIR", str(run_dir))
    monkeypatch.setenv("INFINI_ROOT", str(binaries.parent))
    monkeypatch.setattr(wrapper, "ROOT", tmp_path)
    monkeypatch.setattr(wrapper.shutil, "which", lambda name: None)
    monkeypatch.setattr(wrapper, "official_compare", lambda: SimpleNamespace(
        parse_timing_line=lambda line, timing: timing.update(seen=True)))
    monkeypatch.setattr(sys, "argv", [str(path), "fixture"])
    with pytest.raises(SystemExit) as exit_info:
        wrapper.main()
    assert exit_info.value.code == 0
    runtime = json.loads((run_dir / "runtime.json").read_text())
    assert str(binaries / "fixture.so") in runtime["native_binaries"]
    timing = json.loads((run_dir / "timing.json").read_text())
    assert timing["returncode"] == 0 and timing["observed_transformer_calls"] == 49
    assert "compute.h3_transformer" in (run_dir / "stdout.log").read_text()
    assert sorted(p.name for p in binaries.iterdir()) == ["fixture.so"]
