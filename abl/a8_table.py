"""A8. Render the results table (README section 4 and abl/RESULTS.md) from abl/out/metrics.json."""
from common import GEN_ROWS, OUT, read_json

LABEL = {"base": "base", "steered": "steered (persona-lab refusal vector)",
         "abliterated_a1": "Qwen3.5-4B-abliterated, attempt 1 (L13, A3 pick)",
         "abliterated_a2": "abliterated, attempt 2 (L23)", "abliterated_a3": "abliterated, attempt 3 (L15)",
         "heretic": "Qwen3.5-4B-heretic"}


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def main():
    m = read_json(OUT / "metrics.json")["variants"]
    lines = ["| model | JBB refusals, non-thinking | JBB refusals, thinking | classifier unsafe (non-thinking / thinking) "
             "| JBB benign refusals | first-token KL vs base | GSM8K (Δ vs base ± SE) |",
             "|---|---|---|---|---|---|---|"]
    for v in GEN_ROWS:
        if v not in m:
            lines.append(f"| {LABEL[v]} | see RESULTS.md | | | | | |")
            continue
        r = m[v]
        gsm = "–" if "gsm8k_acc" not in r else f"{100 * r['gsm8k_acc']:.1f}%"
        if v != "base" and "gsm8k_diff_pts" in r:
            gsm += f" ({r['gsm8k_diff_pts']:+.1f} ± {r['gsm8k_diff_se_pts']:.1f})"
        if v.startswith("abliterated") and "gsm8k_acc" not in r:
            r = dict(r, gsm8k_note="not run")
        kl = "0 (reference)" if v == "base" else ("–" if "kl_mean" not in r else f"{r['kl_mean']:.3f}")
        lines.append(f"| {LABEL[v]} | {r.get('jbb_nothink_refusals', '–')}/100 | {r.get('jbb_think_refusals', '–')}/100 | "
                     f"{pct(r.get('jbb_nothink_unsafe_rate'))} / {pct(r.get('jbb_think_unsafe_rate'))} | "
                     f"{r.get('jbb_benign_refusals', '–')}/100 | {kl} | {gsm} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
