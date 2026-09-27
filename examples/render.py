"""Runs each page's script (tutorial_NN.py or example_NN.py) cell by cell and writes its README.md.

The module docstring is the opening text. Cells start with `# %%` (code), `# %% [markdown]` (text written as `# `
comments) or `# %% [hidden]` (run, not shown). Each code cell is shown collapsed, followed by what it printed and
the figures it saved.

`render.py topics/07-declustering` renders one page, no argument renders both galleries, and `--index` only writes
examples/README.md from the scripts, without running them.
"""

import ast
import contextlib
import io
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
GALLERIES = ("tutorials", "topics")
SECTIONS = {
    1: "Data",
    14: "Transforms",
    18: "Variography",
    22: "Estimation",
    32: "Change of support",
    36: "Simulation",
    44: "Validation",
    49: "Modeling",
    57: "I/O and scale",
    61: "More methods",
}
DATASETS = {
    "walker_lake": "Walker Lake",
    "walker_lake_exhaustive": "Walker Lake",
    "jura": "Jura",
    "geomet": "Porphyry geometallurgy",
    "drillhole_tables": "Drillholes (legacy)",
    "drillholes": "Drillholes (legacy)",
}
STOPLIST = {"datasets", "plot", "plot3d", "read_csv"}
LINK = re.compile(r"\]\((?!https?:|#)([^)\s]+)\)")
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
    parts = [ast.get_docstring(ast.parse(script.read_text(encoding="utf-8")))]
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


SCRIPT = {"tutorials": "tutorial_*.py", "topics": "example_*.py"}


def scripts(gallery: str) -> list[Path]:
    return sorted((ROOT / gallery).glob(f"*/{SCRIPT[gallery]}"))


def dataset(name: str) -> str:
    name = name.replace("-", "_")
    return DATASETS.get(name, name.replace("_", " ").capitalize())


def summary(script: Path) -> tuple[int, str, str, str, str]:
    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source)
    number, title = ast.get_docstring(tree).splitlines()[0].removeprefix("# ").split(". ", 1)
    datasets, covers = {}, {}
    for node in sorted(
        (n for n in ast.walk(tree) if isinstance(n, (ast.Attribute, ast.Call))),
        key=lambda n: (n.lineno, n.col_offset),
    ):
        if isinstance(node, ast.Call):
            func = node.func
            if (getattr(func, "attr", None) or getattr(func, "id", None)) == "fetch" and node.args:
                path = node.args[0]
                path = path.values[0] if isinstance(path, ast.JoinedStr) else path
                datasets[dataset(path.value.split("/")[2])] = None
            continue
        base = node.value
        if isinstance(base, ast.Name) and base.id == "cs":
            covers[node.attr] = None
        elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and base.value.id == "cs":
            if base.attr == "datasets" and node.attr != "fetch":
                datasets[dataset(node.attr)] = None
            elif base.attr in ("plot", "plot3d"):
                covers[f"{base.attr}.{node.attr}"] = None
    covers = ", ".join(f"`{name}`" for name in covers if name not in STOPLIST)
    steps = "; ".join(line[5:].strip() for line in source.splitlines() if line.startswith("# ## "))
    return int(number), title, ", ".join(datasets), covers, steps


def broken_links() -> list[str]:
    broken = []
    for gallery in GALLERIES:
        for script in scripts(gallery):
            for target in LINK.findall(script.read_text(encoding="utf-8")):
                if not (script.parent / target).parent.is_dir():
                    broken.append(f"{script.relative_to(ROOT).as_posix()}: {target}")
    return broken


def index() -> str:
    lines = [
        "# Examples",
        "",
        "Worked examples with the Python API on open datasets from [gstvschlz/datasets](https://github.com/gstvschlz/datasets).",
        "Tutorials follow one deposit from data to a result; topics show one feature each.",
        "",
        "## Tutorials",
        "",
        "| # | Tutorial | Dataset | Steps |",
        "|---|---|---|---|",
    ]
    for script in scripts("tutorials"):
        number, title, datasets, _, steps = summary(script)
        link = script.parent.relative_to(ROOT).as_posix()
        lines.append(f"| {number} | [{title}]({link}/README.md) | {datasets} | {steps} |")
    lines += ["", "## Topics", "", "| # | Topic | Dataset | Covers |", "|---|---|---|---|"]
    section = None
    for script in scripts("topics"):
        number, title, datasets, covers, _ = summary(script)
        heading = SECTIONS[max(n for n in SECTIONS if n <= number)]
        if heading != section:
            section = heading
            lines.append(f"| | **{heading}** | | |")
        link = script.parent.relative_to(ROOT).as_posix()
        lines.append(f"| {number} | [{title}]({link}/README.md) | {datasets} | {covers} |")
    lines += [
        "",
        "Each page alternates text, collapsed Python and its results. `mise run examples` reruns every script",
        "and rewrites the pages; `cs.datasets` downloads the data once and caches it. `render.py --index` writes this file.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    if "--index" in sys.argv[1:]:
        if broken := broken_links():
            sys.exit("links to missing folders:\n" + "\n".join(broken))
        (ROOT / "README.md").write_text(index(), encoding="utf-8", newline="\n")
        print("wrote README.md")
        sys.exit()
    pages = sys.argv[1:] or [p.parent.relative_to(ROOT).as_posix() for g in GALLERIES for p in scripts(g)]
    for page in pages:
        (script,) = (ROOT / page).glob(SCRIPT[page.split("/")[0]])
        (script.parent / "README.md").write_text(render(script), encoding="utf-8", newline="\n")
        print(f"rendered {page}")
