"""A1. Build fixed data splits in abl/data/ (gitignored). Commit this script, not the data.

Sources
  harmful train/val : Arditi et al. released splits (github.com/andyrdt/refusal_direction, dataset/splits)
  harmless train/val/KL : tatsu-lab/alpaca, instructions with empty input
  eval : JailbreakBench/JBB-Behaviors harmful (100) + benign (100); openai/gsm8k test (1,319)
Overlap with the eval sets is removed from train/val (normalized exact match + any shared 8-gram).
Refuses to overwrite existing splits: data is fixed once written.
"""
import json
import random
import re
import urllib.request

import pandas as pd
from huggingface_hub import hf_hub_download

from common import DATA, OUT, write_json

ARDITI = "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/{}.json"
SEED = 0


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def grams(s, n=8):
    w = norm(s).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def arditi(name):
    with urllib.request.urlopen(ARDITI.format(name), timeout=60) as r:
        return [d["instruction"] for d in json.load(r)]


def dedupe(items):
    seen, out = set(), []
    for s in items:
        if norm(s) not in seen:
            seen.add(norm(s))
            out.append(s)
    return out


def main():
    names = ["harmful_train", "harmful_val", "harmless_train", "harmless_val", "harmless_kl",
             "jbb_harmful", "jbb_benign", "gsm8k_test"]
    if any((DATA / f"{n}.json").exists() for n in names):
        raise SystemExit("abl/data already has splits; they are fixed once written. Delete by hand only if you mean it.")

    jbb_h = pd.read_csv(hf_hub_download("JailbreakBench/JBB-Behaviors", "data/harmful-behaviors.csv", repo_type="dataset"))
    jbb_b = pd.read_csv(hf_hub_download("JailbreakBench/JBB-Behaviors", "data/benign-behaviors.csv", repo_type="dataset"))
    jbb_harmful = [dict(goal=g, category=c, behavior=b) for g, c, b in zip(jbb_h.Goal, jbb_h.Category, jbb_h.Behavior)]
    jbb_benign = [dict(goal=g, category=c, behavior=b) for g, c, b in zip(jbb_b.Goal, jbb_b.Category, jbb_b.Behavior)]
    assert len(jbb_harmful) == 100 and len(jbb_benign) == 100

    gsm = pd.read_parquet(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset"))
    gsm8k = [dict(question=q, answer=a, target=float(a.split("####")[-1].strip().replace(",", "")))
             for q, a in zip(gsm.question, gsm.answer)]
    assert len(gsm8k) == 1319

    alp = pd.read_parquet(hf_hub_download("tatsu-lab/alpaca", "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
                                          repo_type="dataset"))
    alpaca = dedupe([i.strip() for i, x in zip(alp.instruction, alp.input) if not str(x).strip() and i.strip()])

    harmful_train_src = dedupe(arditi("harmful_train"))
    harmful_val_src = dedupe(arditi("harmful_val"))

    eval_texts = [d["goal"] for d in jbb_harmful + jbb_benign] + [d["question"] for d in gsm8k]
    eval_norm = {norm(t) for t in eval_texts}
    eval_grams = set().union(*(grams(t) for t in eval_texts))

    def clean(items, label, log):
        exact = [s for s in items if norm(s) in eval_norm]
        rest = [s for s in items if norm(s) not in eval_norm]
        gram = [s for s in rest if grams(s) & eval_grams]
        kept = [s for s in rest if not (grams(s) & eval_grams)]
        log[label] = dict(source=len(items), removed_exact=len(exact), removed_8gram=len(gram), kept=len(kept))
        return kept

    log = {}
    rng = random.Random(SEED)
    h_tr = clean(harmful_train_src, "harmful_train_pool", log)
    h_va = clean(harmful_val_src, "harmful_val_pool", log)
    h_va = [s for s in h_va if norm(s) not in {norm(t) for t in h_tr}]
    rng.shuffle(h_tr)
    rng.shuffle(h_va)
    if len(h_tr) < 128 or len(h_va) < 32:
        raise SystemExit(f"not enough harmful prompts after filtering: {len(h_tr)} train, {len(h_va)} val")

    a = clean(alpaca, "alpaca_pool", log)
    rng.shuffle(a)
    harmless_train, harmless_val, harmless_kl = a[:128], a[128:160], a[160:260]

    splits = dict(harmful_train=h_tr[:128], harmful_val=h_va[:32], harmless_train=harmless_train,
                  harmless_val=harmless_val, harmless_kl=harmless_kl, jbb_harmful=jbb_harmful,
                  jbb_benign=jbb_benign, gsm8k_test=gsm8k)
    # Sanity: the instruction splits are disjoint from each other and from eval.
    sets = {k: {norm(s) for s in v} for k, v in splits.items() if isinstance(v[0], str)}
    keys = list(sets)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not (sets[keys[i]] & sets[keys[j]]), (keys[i], keys[j])
        assert not (sets[keys[i]] & eval_norm), keys[i]

    for k, v in splits.items():
        write_json(DATA / f"{k}.json", v)
    manifest = dict(seed=SEED, sizes={k: len(v) for k, v in splits.items()}, overlap_removal=log,
                    method="normalized exact match (lowercase, alnum only) or any shared word 8-gram vs "
                           "JBB harmful+benign goals and GSM8K test questions",
                    sources=dict(harmful="andyrdt/refusal_direction dataset/splits harmful_{train,val}.json",
                                 harmless="tatsu-lab/alpaca (empty input only)",
                                 eval="JailbreakBench/JBB-Behaviors data/{harmful,benign}-behaviors.csv; openai/gsm8k main test"))
    write_json(OUT / "data_manifest.json", manifest)
    print(json.dumps(manifest["sizes"]))
    print(json.dumps(log))


if __name__ == "__main__":
    main()
