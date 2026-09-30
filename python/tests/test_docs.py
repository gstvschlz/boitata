import re
from pathlib import Path

import boitata as bt
import pytest

API = Path(__file__).parents[2] / "docs" / "api"


@pytest.mark.skipif(not API.is_dir(), reason="docs/api not present")
def test_every_public_name_has_an_api_entry():
    documented = {
        name for page in API.glob("*.md") for name in re.findall(r"::: boitata\.(\w+)", page.read_text())
    }
    assert set(bt.__all__) - documented == set()


def test_captured_progress_becomes_one_bar_and_text_stays_one_block():
    hooks_path = Path(__file__).parents[2] / "docs" / "hooks.py"
    if not hooks_path.is_file():
        pytest.skip("docs/hooks.py not present")
    import importlib.util

    spec = importlib.util.spec_from_file_location("hooks", hooks_path)
    hooks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hooks)
    captured = "  0%|          | 0/470 [00:00<?, ?it/s]\n100%|##########| 470/470 [00:01<00:00, 900it/s]\n\nmean 435\nCV 0.69\n"
    out = hooks.OUTPUT.sub(hooks.progress_bars, f"{hooks.FENCE}\n{captured}```")
    assert out.count('bt-progress"') == 1 and "470/470 · 00:01" in out and "it/s" not in out
    assert out.count(hooks.FENCE) == 1 and "mean 435\nCV 0.69" in out
