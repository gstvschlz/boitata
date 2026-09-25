"""Runs each example.py cell by cell and writes its README.md.

Cells start with `# %%` (code), `# %% [markdown]` (text written as `# `
comments) or `# %% [hidden]` (run, not shown). Each code cell is shown collapsed, followed by what it printed and
the figures it saved.
"""

import contextlib
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import common


def cells(source: str):
    kind, lines = None, []
    for line in source.splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                yield kind, "\n".join(lines).strip("\n")
            kind = "markdown" if "[markdown]" in line else "hidden" if "[hidden]" in line else "code"
            lines = []
        elif kind is not None:
            lines.append(line)
    if kind is not None:
        yield kind, "\n".join(lines).strip("\n")


def render(script: Path) -> str:
    namespace = {"__file__": str(script), "__name__": "__main__"}
    parts = []
    for kind, body in cells(script.read_text(encoding="utf-8")):
        if kind == "markdown":
            parts.append(
                "\n".join(
                    line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines()
                )
            )
            continue
        common.SAVED.clear()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(body, str(script), "exec"), namespace)  # noqa: S102
        if kind == "hidden":
            continue
        block = f"<details><summary>Python</summary>\n\n```python\n{body}\n```\n\n</details>"
        printed = out.getvalue().rstrip()
        if printed:
            block += f"\n\n```text\n{printed}\n```"
        for figure in common.SAVED:
            block += f"\n\n![{figure.stem}]({figure.name})"
        parts.append(block)
    parts.append(f"Full script: [`{script.name}`]({script.name})")
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    chapters = sys.argv[1:] or sorted(p.name for p in ROOT.iterdir() if (p / "example.py").exists())
    for chapter in chapters:
        script = ROOT / chapter / "example.py"
        (script.parent / "README.md").write_text(render(script), encoding="utf-8", newline="\n")
        print(f"rendered {chapter}")
