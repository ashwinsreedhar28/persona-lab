"""Read-only split audit. Reports hashes and overlap indices, never prompt text.

    python tools/audit_splits.py --data abl/data --out /tmp/persona-split-audit.json
"""

from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
import json
from pathlib import Path

from research_checks import file_hash, split_overlap


def audit(data):
    files = sorted(Path(data).glob("*.json"))
    if not files:
        raise ValueError("no split JSON files found")
    splits = {}
    for path in files:
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            continue
        splits[path.stem] = [r if isinstance(r, str) else
                             r["goal"] if "goal" in r else
                             r["question"] if "question" in r else r["instruction"] for r in rows]
    result = split_overlap(splits)
    result["sha256"] = {p.name: file_hash(p) for p in files}
    result["policy"] = "diagnostic only; fixed splits unchanged; review indices locally"
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("abl/data"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.data)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(sizes=result["sizes"], duplicates=result["duplicates"],
                         overlaps=[dict(a=p["a"], b=p["b"], exact=len(p["exact"]),
                                        shared_8gram=len(p["shared_8gram"]))
                                   for p in result["pairs"] if p["exact"] or p["shared_8gram"]])))
