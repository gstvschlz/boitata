import re
from pathlib import Path

import boitata as bt
import pytest

API = Path(__file__).parents[2] / "docs" / "api"
pytestmark = pytest.mark.skipif(not API.is_dir(), reason="docs/api not present")


def test_every_public_name_has_an_api_entry():
    documented = {
        name for page in API.glob("*.md") for name in re.findall(r"::: boitata\.(\w+)", page.read_text())
    }
    assert set(bt.__all__) - documented == set()
