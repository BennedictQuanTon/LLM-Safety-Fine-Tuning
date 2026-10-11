"""Build Kaggle-ready .ipynb files from percent-format sources in notebooks/src/.

Usage: python3 tools/build_notebooks.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "notebooks" / "src"
OUT_DIR = ROOT / "notebooks"


def parse_cells(text: str):
    cells, kind, lines = [], None, []

    def flush():
        while lines and not lines[-1].strip():
            lines.pop()
        if kind is not None and lines:
            cells.append((kind, list(lines)))

    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped in ("# %%", "# %% [markdown]"):
            flush()
            kind, lines = ("markdown" if stripped.endswith("[markdown]") else "code"), []
            continue
        if kind == "markdown":
            lines.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
        elif kind == "code":
            if not lines and not line.strip():
                continue
            lines.append(line)
    flush()
    return cells


def to_notebook(cells):
    nb_cells = []
    for kind, lines in cells:
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    for src in sorted(SRC_DIR.glob("*.py")):
        cells = parse_cells(src.read_text(encoding="utf-8"))
        out = OUT_DIR / f"{src.stem}.ipynb"
        out.write_text(json.dumps(to_notebook(cells), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{out.relative_to(ROOT)}: {len(cells)} cells")


if __name__ == "__main__":
    main()
