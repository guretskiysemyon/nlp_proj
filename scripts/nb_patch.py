"""Small helpers to patch .ipynb files by cell content (not by index), so a patch survives cell shifts.

    nb = load(path)
    i = find(nb, "SMF_PATH = RUN_DIR")                 # the one cell whose source contains the anchor
    set_source(nb, i, new_src)                         # replace a cell (outputs cleared)
    insert_after(nb, i, [code(src), markdown(md)])     # new cells
    replace_in(nb, i, old, new)                        # exact substring replacement inside one cell
    save(nb, path)

Every function raises if an anchor is missing or ambiguous, so a patch never lands in the wrong cell.
Idempotence is the caller's job: check for a marker (e.g. `has(nb, "JUDGE_FORMAT")`) before patching.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path


def load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(nb: dict, path) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def src(cell: dict) -> str:
    return "".join(cell["source"])


def _lines(text: str) -> list:
    out = text.splitlines(keepends=True)
    return out


def find_all(nb: dict, anchor: str, cell_type: str | None = None) -> list:
    return [i for i, c in enumerate(nb["cells"])
            if anchor in src(c) and (cell_type is None or c["cell_type"] == cell_type)]


def find(nb: dict, anchor: str, cell_type: str | None = None) -> int:
    hits = find_all(nb, anchor, cell_type)
    if len(hits) != 1:
        raise KeyError(f"anchor {anchor!r} matches {len(hits)} cells {hits}, expected exactly 1")
    return hits[0]


def has(nb: dict, anchor: str) -> bool:
    return bool(find_all(nb, anchor))


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def code(text: str, tags=None) -> dict:
    c = {"cell_type": "code", "execution_count": None, "id": _new_id(), "metadata": {}, "outputs": [],
         "source": _lines(text)}
    if tags:
        c["metadata"]["tags"] = list(tags)
    return c


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "id": _new_id(), "metadata": {}, "source": _lines(text)}


def set_source(nb: dict, i: int, text: str) -> None:
    c = nb["cells"][i]
    c["source"] = _lines(text)
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None


def replace_in(nb: dict, i: int, old: str, new: str, count: int = 1) -> None:
    s = src(nb["cells"][i])
    n = s.count(old)
    if n != count:
        raise KeyError(f"cell {i}: {old[:60]!r} occurs {n} times, expected {count}")
    set_source(nb, i, s.replace(old, new))


def insert_after(nb: dict, i: int, cells: list) -> None:
    nb["cells"][i + 1:i + 1] = cells


def insert_before(nb: dict, i: int, cells: list) -> None:
    nb["cells"][i:i] = cells


def delete(nb: dict, i: int) -> None:
    del nb["cells"][i]


def clear_outputs(nb: dict) -> None:
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
