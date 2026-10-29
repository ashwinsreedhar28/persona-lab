"""A8. Render the results table (README section 4 and abl/RESULTS.md) from abl/out/metrics.json.

    python abl/a8_table.py                                         # all rows (RESULTS.md)
    python abl/a8_table.py --rows base steered abliterated_a1 heretic   # the README table
"""
import argparse

from common import GEN_ROWS, OUT, read_json

LABEL = {"base": "base", "steered": "steered (persona-lab refusal vector)",
         "abliterated_a1": "Qwen3.5-4B-abliterated (attempt 1: L13, A3's val pick)",
         "abliterated_a2": "attempt 2: L23 (exploratory)", "abliterated_a3": "attempt 3: L15 (exploratory)",
         "heretic": "Qwen3.5-4B-heretic"}


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def frac(r, task):
    return "not run" if f"{task}_refusals" not in r else f"{r[f'{task}_refusals']}/{r[f'{task}_n']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=list(GEN_ROWS), choices=GEN_ROWS)
    rows = ap.parse_args().rows
    m = read_json(OUT / "metrics.json")["variants"]
    heretic_deferred = (OUT / "heretic_timing.json").exists() and \
        read_json(OUT / "heretic_timing.json").get("decision") == "defer"
    lines = ["| model | JBB refusals, non-thinking | JBB refusals, thinking | classifier unsafe (non-thinking / thinking) "
             "| JBB benign refusals | first-token KL vs base | GSM8K (Δ vs base ± SE) |",
             "|---|---|---|---|---|---|---|"]
    for v in rows:
        if v not in m:
            note = "deferred to Phase B pod (see RESULTS.md)" if v == "heretic" and heretic_deferred else "not run"
            lines.append(f"| {LABEL[v]} | {note} | | | | | |")
            continue
        r = m[v]
        gsm = "not run" if "gsm8k_acc" not in r else f"{100 * r['gsm8k_acc']:.1f}%"
        if v != "base" and "gsm8k_diff_pts" in r:
            gsm += f" ({r['gsm8k_diff_pts']:+.1f} ± {r['gsm8k_diff_se_pts']:.1f})"
        kl = "0 (reference)" if v == "base" else ("–" if "kl_mean" not in r else f"{r['kl_mean']:.3f}")
        lines.append(f"| {LABEL[v]} | {frac(r, 'jbb_nothink')} | {frac(r, 'jbb_think')} | "
                     f"{pct(r.get('jbb_nothink_unsafe_rate'))} / {pct(r.get('jbb_think_unsafe_rate'))} | "
                     f"{frac(r, 'jbb_benign')} | {kl} | {gsm} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
