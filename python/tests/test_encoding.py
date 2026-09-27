from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BROKEN = tuple(s.encode("utf-8").decode("cp1252") for s in ("—", "√", "σ", "²", "³", "×", "±", "°"))


def text_files():
    for folder in ("crates", "python", "examples", "docs"):
        base = ROOT / folder
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if (
                path.suffix in (".rs", ".py", ".pyi", ".md", ".txt")
                and "target" not in path.parts
                and path.name != "test_encoding.py"
            ):
                yield path


def test_no_double_encoded_text():
    if not (ROOT / "crates").exists():
        pytest.skip("source tree not available")
    bad = [
        f"{path.relative_to(ROOT)}: {token}"
        for path in text_files()
        for token in BROKEN
        if token in path.read_text(encoding="utf-8", errors="replace")
    ]
    assert not bad, "double-encoded UTF-8:\n" + "\n".join(bad)
