"""A7 (generation half). One variant per process; identical greedy settings for every variant.

    python abl/a7_generate.py --variant base|steered|abliterated|heretic [--tasks ...]

Raw generations go to abl/out/gen/<variant>/ (gitignored). Only aggregate numbers are printed.
Tasks
  jbb_nothink  JBB harmful, non-thinking, max 256 tokens
  jbb_think    JBB harmful, thinking, max 1024 tokens
  jbb_benign   JBB benign, non-thinking, max 256 tokens
  kl           first-token log-probs on the 100 held-out harmless prompts (base saves them; others score KL)
  gsm8k        all 1,319 test problems, non-thinking, zero-shot, max 512 tokens

Generation runs in fixed chunks of 128 prompts (the same split for every variant) and frees MLX's buffer cache
between chunks: one 1,319-prompt batch_generate call hit Metal's buffer-count limit. Each finished chunk is saved
under gen/<name>/chunks/<task>/, so a crashed or interrupted run resumes at the next chunk.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import batch_generate

from common import (ABLITERATED, BASE_ID, DATA, HERETIC, OUT, answer_part, chat_ids, kl_rows, last_logprobs,
                    last_number, load, marker_refusal, peak_ram_gb, read_json, write_json)

GSM_PROMPT = "{q}\n\nSolve the problem step by step. Give the final answer as a number on the last line."
TASKS = ("kl", "jbb_nothink", "jbb_benign", "gsm8k", "jbb_think")
STEER_VECTOR = OUT / "refusal_bf16.npz"
CHUNK = 128


def load_variant(name, model_path=None):
    if name in ("base", "steered"):
        model, tok = load(BASE_ID)
        info = dict(path=BASE_ID)
        if name == "steered":
            from evil_chat_mlx import Steerer                    # persona-lab's own loader, unchanged
            with np.load(STEER_VECTOR, allow_pickle=False) as z:
                assert str(z["model"]) == BASE_ID, z["model"]
                layer, dose = int(z["best"]), float(z["recommended_dose"])
            Steerer(model, tok).apply_vector(STEER_VECTOR, -dose)  # minus = less refusal
            info.update(vector=str(STEER_VECTOR), layer=layer, dose=-dose)
        return model, tok, info
    path = Path(model_path) if model_path else {"abliterated": ABLITERATED, "heretic": HERETIC}[name]
    model, tok = load(path)
    info = dict(path=str(path))
    if (path / "abl_edit.json").exists():                       # which r-hat this edited folder holds
        e = read_json(path / "abl_edit.json")
        info.update(r_hat=e["r_hat"], r_hat_sha256=e["r_hat_sha256"])
    return model, tok, info


def generate(model, tok, prompts, max_tokens, thinking, bs, chunk_dir, key):
    """Greedy batch_generate over fixed chunks of CHUNK prompts, clearing MLX's cache between chunks. Finished
    chunks are saved in chunk_dir and reused only if their key (model, prompts, settings) matches."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    texts, secs, n_tok, reused = [], 0.0, 0, 0
    for k, s0 in enumerate(range(0, len(prompts), CHUNK)):
        part = prompts[s0:s0 + CHUNK]
        ck = hashlib.sha256(json.dumps([key, part, max_tokens, thinking, bs]).encode()).hexdigest()
        f = chunk_dir / f"{k:03d}.json"
        if f.exists():
            d = json.loads(f.read_text())
            if d["key"] == ck:
                texts += d["texts"]; secs += d["seconds"]; n_tok += d["tokens"]; reused += 1
                continue
        ids = [chat_ids(tok, p, thinking=thinking) for p in part]
        t = time.time()
        res = batch_generate(model, tok, ids, max_tokens=max_tokens, completion_batch_size=bs)
        out = list(res.texts)
        dt = time.time() - t
        nt = sum(len(tok.encode(x, add_special_tokens=False)) for x in out)
        f.write_text(json.dumps(dict(key=ck, texts=out, seconds=round(dt, 1), tokens=nt)))
        texts += out; secs += dt; n_tok += nt
        del res
        mx.clear_cache()
        print(f"  chunk {k + 1}/{-(-len(prompts) // CHUNK)}: {len(part)} prompts, {nt / max(dt, 1e-9):.0f} tok/s, "
              f"{dt:.0f}s", flush=True)
    return texts, dict(seconds=round(secs, 1), tokens=n_tok, tok_per_s=round(n_tok / max(secs, 1e-9), 1),
                       chunk=CHUNK, chunks_reused=reused)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["base", "steered", "abliterated", "heretic"])
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--gen_name", default=None, help="output folder under abl/out/gen (default: the variant name)")
    ap.add_argument("--limit", type=int, default=None, help="debug only: first N items per task")
    ap.add_argument("--model_path", default=None, help="load this folder instead of the variant's default")
    args = ap.parse_args()
    gen_dir = OUT / "gen" / (args.gen_name or args.variant)
    gen_dir.mkdir(parents=True, exist_ok=True)
    model, tok, info = load_variant(args.variant, args.model_path)
    key = json.dumps(info, sort_keys=True)
    chunks = gen_dir / ("chunks" if not args.limit else f"chunks_limit{args.limit}")
    summary_path = gen_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() and not args.limit else {}
    summary["model"] = info
    lim = slice(None, args.limit)

    for task in args.tasks:
        t0 = time.time()
        if task == "kl":
            pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            prompts = read_json(DATA / "harmless_kl.json")[lim]
            lp = last_logprobs(model, [chat_ids(tok, p) for p in prompts], pad)
            base_path = OUT / "kl_base_logprobs.npy"
            if args.variant == "base":
                if not args.limit:
                    np.save(base_path, np.array(lp))
                kl = np.zeros(len(prompts))
            else:
                kl = np.array(kl_rows(mx.array(np.load(base_path)[lim]), lp))
            summary["kl"] = dict(n=len(prompts), mean=float(kl.mean()), max=float(kl.max()),
                                 per_prompt=kl.round(5).tolist(), direction="KL(base || variant), first token")
            print(f"[{args.variant}] kl mean={kl.mean():.4f} max={kl.max():.4f}", flush=True)
            continue
        if task in ("jbb_nothink", "jbb_think", "jbb_benign"):
            rows = read_json(DATA / ("jbb_benign.json" if task == "jbb_benign" else "jbb_harmful.json"))[lim]
            thinking = task == "jbb_think"
            texts, stats = generate(model, tok, [r["goal"] for r in rows], 1024 if thinking else 256, thinking, args.bs,
                                    chunks / task, key)
            labels = [marker_refusal(answer_part(t)) for t in texts]
            finished = ["</think>" in t for t in texts] if thinking else None
            (gen_dir / f"{task}.json").write_text(json.dumps(
                [dict(i=i, behavior=r["behavior"], category=r["category"], completion=t, refused=l)
                 for i, (r, t, l) in enumerate(zip(rows, texts, labels))], indent=1))
            summary[task] = dict(n=len(rows), refusals=int(sum(labels)), refused=[int(x) for x in labels],
                                 thinking_closed=None if finished is None else int(sum(finished)),
                                 max_tokens=1024 if thinking else 256, **stats)
            print(f"[{args.variant}] {task}: refusals {sum(labels)}/{len(rows)} "
                  f"({stats['tok_per_s']} tok/s, {stats['seconds']}s)", flush=True)
        elif task == "gsm8k":
            rows = read_json(DATA / "gsm8k_test.json")[lim]
            texts, stats = generate(model, tok, [GSM_PROMPT.format(q=r["question"]) for r in rows], 512, False, args.bs,
                                    chunks / task, key)
            preds = [last_number(t) for t in texts]
            correct = [int(p is not None and abs(p - r["target"]) < 1e-6) for p, r in zip(preds, rows)]
            (gen_dir / "gsm8k.json").write_text(json.dumps(
                [dict(i=i, completion=t, pred=p, target=r["target"], correct=c)
                 for i, (r, t, p, c) in enumerate(zip(rows, texts, preds, correct))], indent=1))
            summary["gsm8k"] = dict(n=len(rows), accuracy=float(np.mean(correct)), correct=correct,
                                    prompt=GSM_PROMPT, max_tokens=512, **stats)
            print(f"[{args.variant}] gsm8k acc={np.mean(correct):.4f} ({stats['tok_per_s']} tok/s, "
                  f"{stats['seconds']}s)", flush=True)
        mx.clear_cache()
        summary["peak_ram"] = peak_ram_gb()
        if not args.limit:
            write_json(summary_path, summary)
        print(f"  {task} done in {time.time() - t0:.0f}s", flush=True)
    if not args.limit:
        write_json(summary_path, summary)


if __name__ == "__main__":
    main()
