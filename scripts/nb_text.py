"""Print the text outputs (and errors) of executed notebooks: quick review of expected outputs."""
import json
import sys

for path in sys.argv[1:]:
    nb = json.load(open(path))
    print(f"##### {path}")
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        for o in c.get("outputs", []):
            if o["output_type"] == "stream":
                print(f"[{i}]", "".join(o["text"]).rstrip())
            elif o["output_type"] == "error":
                print(f"[{i}] ERROR", o["ename"], o["evalue"])
            elif "text/plain" in o.get("data", {}) and "image/png" not in o["data"]:
                print(f"[{i}]", "".join(o["data"]["text/plain"]).rstrip()[:500])
