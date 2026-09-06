"""Keep optional PyTorch conversion checks outside the lightweight process."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def test_checkpoint_velocity_numeric_and_native_call_sites_in_isolated_process(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch CPU tensors are required for the isolated conversion cases")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(Path(__file__).with_name("checkpoint_velocity_cases.py")),
            f"--junitxml={tmp_path / 'checkpoint-velocity-cases.xml'}",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "29 passed" in result.stdout
