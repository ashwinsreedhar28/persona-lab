"""C5 helper (runs on the pod before bring-back). Two jobs:
1. Guard: /workspace/out may hold only .json files, and no JSON may contain a numeric array of length >= d_model (2304)
   (that would be a direction or an activation). Exits non-zero if anything fails.
2. summary.json: the headline numbers for RESULTS.md.
"""
import json
import sys
from pathlib import Path

OUT = Path("/workspace/out")
D = 2304


def long_numeric_lists(x, path="$"):
    if isinstance(x, list):
        if len(x) >= D and all(isinstance(v, (int, float)) for v in x[:D]):
            yield path, len(x)
        for i, v in enumerate(x):
            yield from long_numeric_lists(v, f"{path}[{i}]")
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from long_numeric_lists(v, f"{path}.{k}")


def rj(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    bad = [p.name for p in OUT.iterdir() if p.is_file() and p.suffix != ".json"]
    for p in sorted(OUT.glob("*.json")):
        hits = list(long_numeric_lists(json.loads(p.read_text())))
        if hits:
            bad.append(f"{p.name}: {hits[:3]}")
    if bad:
        print("GUARD FAIL:", bad)
        sys.exit(1)
    print("GUARD OK:", sorted(p.name for p in OUT.glob("*.json")))

    w, st, sel, att, proj, kl = (rj(n) for n in ("writers.json", "directions_stats.json", "selection.json",
                                                  "attribution.json", "projection.json", "kl.json"))
    s = {}
    if w:
        s["writers"] = dict(n=w["n_writers"], by_kind=w["writers_by_kind"], scan_match=w["generic_scan_matches_structural"],
                            bias=w["any_writer_bias"], tied=w["tie_word_embeddings"],
                            method_check_kl=w["method_check"]["kl_hook_vs_weight_edit"])
    if sel:
        s["selected"] = sel.get("selected")
        s["refusal_tokens"] = sel.get("refusal_tokens")
    if proj:
        s["max_ratio_full_val"] = proj["max_ratio_full_val"]
        s["max_ratio_noshared_val"] = proj["max_ratio_noshared_val"]
    if kl:
        s["kl_mean"] = dict(noshared=kl["noshared"]["mean"], full=kl["full"]["mean"])
        s["equivalence_kl"] = kl["equivalence_full_edit_vs_hook"]
    if att:
        d = att["scopes"]["pos"]["diff_harmful_minus_harmless"]
        s["attribution_pos_diff_totals"] = dict(
            embed=d["embed"], attn=sum(d["attn"]), dense_l0=d["dense_mlp_l0"],
            routed=sum(x for x in d["routed"] if x is not None), shared=sum(x for x in d["shared"] if x is not None))
    (OUT / "summary.json").write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
