"""Writes notebooks/learn_NN.ipynb from each learn/NN-*/learn_NN.py, for Colab.

The docstring and markdown cells become text: admonitions turn into quotes, `--8<--` figures into SVG attachments
styled like the docs, math delimiters into `$`, and relative links into links to the docs site. Hidden cells are
dropped and a setup cell installs boitata and fetches common.py; a script using `bt.plot3d` also gets pyvista with its
standalone HTML viewer, which works in Colab. pyvista 0.49 imports `IPython.core.guarded_eval`, which Colab's IPython
lacks, so the cell holds pyvista below 0.49.
"""

import ast
import base64
import json
import posixpath
import re
from pathlib import Path

from render import cells

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://gstvschlz.github.io/boitata/"
RAW = "https://raw.githubusercontent.com/gstvschlz/boitata/main/"
SETUP = f"""import importlib.util
import urllib.request

if importlib.util.find_spec("boitata") is None:
    %pip install -q "boitata[plot]>=0.3"
if importlib.util.find_spec("common") is None:
    urllib.request.urlretrieve("{RAW}examples/common.py", "common.py")"""
SETUP_3D = (
    SETUP.replace('"boitata[plot]>=0.3"', '"boitata[all]>=0.4.1" "pyvista[jupyter]<0.49"')
    + """

import pyvista as pv

pv.set_jupyter_backend("html")"""
)
ADMONITION = re.compile(r'^(?:!!!|\?\?\?\+?) (\w+)(?: "(.*)")?$')
FIGURE = re.compile(
    r'<figure[^>]*>\s*--8<-- "([^"]+)"\s*<figcaption>(.*?)</figcaption>\s*</figure>', re.DOTALL
)
LINK = re.compile(r"\]\((?!https?:|#)([^)\s]+?)\.md(#[^)\s]*)?\)")
MATH = {r"\(": "$", r"\)": "$", r"\[": "$$", r"\]": "$$"}
CSS = (ROOT / "docs/assets/components.css").read_text(encoding="utf-8")
TOKENS = dict(re.findall(r"(--bt-[\w-]+):\s*([^;]+);", CSS.split("}")[0]))
STYLE = re.sub(
    r"var\((--[\w-]+)\)",
    lambda m: TOKENS.get(m[1], "sans-serif"),
    "\n".join(line for line in CSS.splitlines() if line.startswith(".bt-svg") and " .a-" not in line),
)


def svg(name: str) -> str:
    text = (ROOT / "docs/snippets" / name).read_text(encoding="utf-8")
    text = (
        text.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1) if "xmlns=" not in text else text
    )
    text = re.sub(r"(<svg[^>]*>)", rf"\1<style>{STYLE}</style>", text, count=1)
    return base64.b64encode(text.encode()).decode()


def quotes(text: str) -> str:
    out, inside = [], False
    for line in text.splitlines():
        if match := ADMONITION.match(line):
            out += [f"> **{match[2] or match[1].capitalize()}**", ">"]
            inside = True
        elif inside and (line.startswith("    ") or not line.strip()):
            out.append(("> " + line[4:]).rstrip())
        else:
            if inside and out[-1] == ">":
                out[-1] = ""
            inside = False
            out.append(line)
    return "\n".join(out)


def markdown(text: str, page: str) -> dict:
    attachments = {}

    def figure(match):
        name = Path(match[1]).name
        attachments[name] = {"image/svg+xml": svg(match[1])}
        caption = re.sub(r"</?b>", "**", " ".join(match[2].split()))
        return f"![{name}](attachment:{name})\n\n{caption}"

    text = FIGURE.sub(figure, quotes(text))
    text = LINK.sub(lambda m: f"]({SITE}{posixpath.normpath(posixpath.join(page, m[1]))}/{m[2] or ''})", text)
    for old, new in MATH.items():
        text = text.replace(old, new)
    cell = {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
    return cell | ({"attachments": attachments} if attachments else {})


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook(script: Path) -> dict:
    page = script.parent.relative_to(ROOT).as_posix()
    source = script.read_text(encoding="utf-8")
    intro = ast.get_docstring(ast.parse(source))
    body = [code(SETUP_3D if "bt.plot3d" in source else SETUP)]
    for kind, text in cells(source):
        if kind == "markdown":
            body.append(
                markdown(
                    "\n".join(
                        line[2:] if line.startswith("# ") else line.lstrip("#") for line in text.splitlines()
                    ),
                    page,
                )
            )
        elif kind == "code":
            body.append(code(text))
    return {
        "cells": [markdown(intro, page), *body],
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


if __name__ == "__main__":
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for script in sorted(ROOT.glob("learn/*/learn_*.py")):
        path = out / f"{script.stem}.ipynb"
        path.write_text(json.dumps(notebook(script), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT).as_posix())
