import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import boitata as bt
import pytest

ROOT = Path(__file__).parents[2]
API = ROOT / "docs" / "api"
HOOKS = ROOT / "docs" / "hooks.py"


@pytest.fixture(scope="module")
def hooks():
    if not HOOKS.is_file():
        pytest.skip("docs/hooks.py not present")
    spec = importlib.util.spec_from_file_location("hooks", HOOKS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not API.is_dir(), reason="docs/api not present")
def test_every_public_name_has_an_api_entry():
    documented = {
        name for page in API.glob("*.md") for name in re.findall(r"::: boitata\.(\w+)", page.read_text())
    }
    assert set(bt.__all__) - documented == set()


def test_captured_progress_becomes_one_bar_and_text_stays_one_block(hooks):
    captured = "  0%|          | 0/470 [00:00<?, ?it/s]\n100%|##########| 470/470 [00:01<00:00, 900it/s]\n\nmean 435\nCV 0.69\n"
    out = hooks.OUTPUT.sub(hooks.progress_bars, f"{hooks.FENCE}\n{captured}```")
    assert out.count('bt-progress"') == 1 and "470/470 · 00:01" in out and "it/s" not in out
    assert out.count(hooks.FENCE) == 1 and "mean 435\nCV 0.69" in out


def test_trail_marks_earlier_chapters_done_and_leaves_missing_pages_unlinked(hooks):
    chapters = [("01-a", "A", True), ("02-b", "B", True), ("03-c", "C", True), ("04-d", "D", False)]
    out = hooks.trail(chapters, "02")
    assert '<li class="done" markdown="span">[1. A](../01-a/learn_01.md)</li>' in out
    assert '<li class="here" markdown="span"><span>2. B</span></li>' in out
    assert "[3. C](../03-c/learn_03.md)" in out and "<span>4. D</span>" in out
    assert [slug for slug, _, _ in hooks.chapters(ROOT)] == hooks.CHAPTERS.split()


def test_learn_page_gets_trail_after_heading_and_search_boost(hooks):
    page = SimpleNamespace(file=SimpleNamespace(src_uri="learn/01-samples-and-support/learn_01.md"), meta={})
    out = hooks.on_page_markdown("<!-- x -->\n\n# Samples\n\nIntro.", page)
    assert out.index("# Samples") < out.index('class="bt-trail"') < out.index("Intro.")
    assert page.meta["search"] == {"boost": 2}
    times = SimpleNamespace(file=SimpleNamespace(src_uri="examples/01-a/01-b/mg_execution_times.md"), meta={})
    hooks.on_page_markdown("# Times", times)
    assert times.meta["search"] == {"exclude": True}


def test_used_in_finds_calls_and_chips_prefer_the_first_pages(hooks):
    pages = [
        ("learn/01-a/learn_01.md", "One", "sk = bt.OrdinaryKriging(v)"),
        (
            "examples/06-k/01-o/example_06_01.md",
            "Two",
            "from x import y\nok.OrdinaryKriging(\nbt.describe(v)",
        ),
        ("examples/03-e/01-d/example_03_01.md", "Three", "df.describe()"),
    ]
    uses = hooks.used_in(["OrdinaryKriging", "describe"], pages)
    assert [t for _, t in uses["OrdinaryKriging"]] == ["One", "Two"] and [t for _, t in uses["describe"]] == [
        "Two"
    ]
    chips = hooks.used_in_chips(uses["OrdinaryKriging"], cap=1)
    assert "[One](../../learn/01-a/learn_01.md)" in chips and "Two" not in chips
    assert hooks.used_in_chips([]) == ""
    scripts = [uri for uri, _, _ in hooks.script_pages(ROOT)]
    assert all(uri.startswith("learn/") for uri in scripts[: sum(u.startswith("learn/") for u in scripts)])


def test_glossary_is_sorted_deduplicated_raw_html(hooks):
    out = hooks.glossary([("sill", "The `plateau`."), ("Nugget", "Short <scale>."), ("sill", "Again.")])
    assert out.startswith("# glossary\n") and out.index(">Nugget<") < out.index(">sill<")
    assert '<dt id="sill">' in out and "<code>plateau</code>" in out and "&lt;scale&gt;" in out
    assert out.count("<dt") == 2 and "Again." not in out
