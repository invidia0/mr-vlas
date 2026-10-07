"""Apply the same text replacements to a lab's jupytext source and its executed notebook's markdown cells,
so narrative fixes do not require re-executing a long lab.  Usage: patch_md.py lab.py replacements.json"""
import json
import sys
from pathlib import Path

src, reps = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
nb = src.with_suffix(".ipynb")
text = src.read_text()
book = json.loads(nb.read_text())
for old, new in reps:
    old_py, new_py = "\n".join("# " + l if l else "#" for l in old.split("\n")), "\n".join("# " + l if l else "#" for l in new.split("\n"))
    if text.count(old_py) != 1:                     # an in-line substring: no comment prefix to add
        old_py, new_py = old, new
    assert text.count(old_py) == 1, f"not found once in {src}: {old[:60]}"
    text = text.replace(old_py, new_py)
    hits = 0
    for cell in book["cells"]:
        s = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and old in s:
            s = s.replace(old, new)
            cell["source"] = s.splitlines(keepends=True)
            hits += 1
    assert hits == 1, f"not found once in {nb}: {old[:60]}"
src.write_text(text)
nb.write_text(json.dumps(book, indent=1, ensure_ascii=False) + "\n")
print("patched", src, nb)
