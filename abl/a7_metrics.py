"""A7 (aggregation). Collect per-variant summaries, judge labels and the mechanical check into
abl/out/metrics.json. GSM8K differences are paired per problem vs base, with their standard error."""
import json

import numpy as np

from common import GEN_ROWS, OUT, read_json, write_json


def main():
    out = dict(variants={}, notes=[])
    base_gsm = None
    p = OUT / "gen" / "base" / "summary.json"
    if p.exists() and "gsm8k" in read_json(p):
        base_gsm = np.array(read_json(p)["gsm8k"]["correct"])
    for v in GEN_ROWS:
        p = OUT / "gen" / v / "summary.json"
        if not p.exists():
            continue
        s = read_json(p)
        j = read_json(OUT / "gen" / v / "judge.json") if (OUT / "gen" / v / "judge.json").exists() else {}
        m = dict(model=s.get("model"))
        for task in ("jbb_nothink", "jbb_think", "jbb_benign"):
            if task in s:
                m[f"{task}_refusals"] = s[task]["refusals"]
                m[f"{task}_n"] = s[task]["n"]
        if "jbb_think" in s:
            m["jbb_think_closed_thinking"] = s["jbb_think"]["thinking_closed"]
        for task in ("jbb_nothink", "jbb_think"):
            if task in j:
                m[f"{task}_unsafe_rate"] = j[task]["unsafe_rate"]
                m[f"{task}_controversial"] = j[task]["controversial"]
                m[f"{task}_unparsed"] = j[task]["unparsed"]
        if j:
            m["classifier"] = j.get("classifier")
        if "kl" in s:
            m["kl_mean"], m["kl_max"], m["kl_n"] = s["kl"]["mean"], s["kl"]["max"], s["kl"]["n"]
        if "gsm8k" in s:
            c = np.array(s["gsm8k"]["correct"])
            m["gsm8k_n"], m["gsm8k_acc"] = int(len(c)), float(c.mean())
            if base_gsm is not None and len(base_gsm) == len(c):
                d = (c - base_gsm) * 100.0
                m["gsm8k_diff_pts"] = float(d.mean())
                m["gsm8k_diff_se_pts"] = float(d.std(ddof=1) / np.sqrt(len(d))) if v != "base" else 0.0
        out["variants"][v] = m
    out["mechanical"] = {}
    for n in (1, 2, 3):
        proj = OUT / f"attempt{n}" / "projection.json"
        if proj.exists():
            pj = read_json(proj)
            out["mechanical"][f"abliterated_a{n}"] = dict(
                max_ratio=pj["max_ratio"], argmax=pj["argmax"], per_layer_ratio=[r["ratio"] for r in pj["rows"]],
                edit_vs_hook_kl=pj["edit_vs_hook_first_token_kl"])
    if (OUT / "attempts.json").exists():
        out["attempts"] = read_json(OUT / "attempts.json")
    # "abliterated" = attempt 1: A3's selection and the model left at models/Qwen3.5-4B-abliterated.
    if "abliterated_a1" in out["variants"]:
        out["variants"]["abliterated"] = dict(out["variants"]["abliterated_a1"], alias_of="abliterated_a1")
    if "abliterated_a1" in out["mechanical"]:
        out["mechanical"]["abliterated"] = out["mechanical"]["abliterated_a1"]
    sel = OUT / "selection.json"
    if sel.exists():
        out["selected_direction"] = read_json(sel)["selected"]
    write_json(OUT / "metrics.json", out)
    print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in m.items() if kk != "model"}
                      for k, m in out["variants"].items()}, indent=1))


if __name__ == "__main__":
    main()
