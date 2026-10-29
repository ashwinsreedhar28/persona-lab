"""Non-interactive driver for the Heretic baseline (A6). Runs inside .venv-heretic; Heretic itself is an
installed AGPL dependency and is not vendored here. Run from abl/out/heretic/ (its checkpoints live there).

    probe : ../../../.venv-heretic/bin/python ../../heretic_run.py --probe -- --model Qwen/Qwen3.5-4B --n-trials 3 ...
    full  : ../../../.venv-heretic/bin/python ../../heretic_run.py --more K --save <dir> -- --model Qwen/Qwen3.5-4B ...

Heretic's interactive prompts are answered by fixed rules:
  resume prompt        -> continue / show previous results
  trial menu (first)   -> run K additional trials (if --more K), then
  trial menu           -> Pareto trial with the fewest refusals among KL <= 0.1 (ties: lower KL);
                          if none has KL <= 0.1, the lowest-KL Pareto trial
  action menu          -> "Save the model to a local folder" once (merged), then exit
Any other prompt (upload, chat, benchmark, passwords) raises: nothing is ever uploaded.
"""
import argparse
import json
import sys
from pathlib import Path

import heretic.main as hm

KL_CAP = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="stop at the trial menu without saving")
    ap.add_argument("--more", type=int, default=0, help="additional trials to run when resuming")
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--pick_log", type=str, default="heretic_pick.json")
    a, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    sys.argv = ["heretic"] + rest
    state = dict(extended=False, picked=False, saved=False)

    def select(msg, choices):
        vals = [getattr(c, "value", c) for c in choices]
        if msg.startswith("How would you like to proceed"):
            return "continue"
        if msg.startswith("Which trial"):
            if a.probe or state["saved"]:
                return ""
            if a.more > 0 and not state["extended"]:
                state["extended"] = True
                return "continue"
            trials = [v for v in vals if hasattr(v, "user_attrs")]
            ok = [t for t in trials if t.user_attrs["kl_divergence"] <= KL_CAP]
            pick = (min(ok, key=lambda t: (t.user_attrs["refusals"], t.user_attrs["kl_divergence"])) if ok
                    else min(trials, key=lambda t: t.user_attrs["kl_divergence"]))
            info = dict(rule=f"fewest refusals among Pareto trials with KL <= {KL_CAP}, else lowest KL",
                        picked=dict(index=pick.user_attrs.get("index"), refusals=pick.user_attrs["refusals"],
                                    kl=pick.user_attrs["kl_divergence"],
                                    n_bad_prompts=pick.user_attrs.get("n_bad_prompts"),
                                    base_refusals=pick.user_attrs.get("base_refusals"),
                                    direction_index=pick.user_attrs.get("direction_index"),
                                    parameters=pick.user_attrs.get("parameters")),
                        pareto=[dict(index=t.user_attrs.get("index"), refusals=t.user_attrs["refusals"],
                                     kl=t.user_attrs["kl_divergence"]) for t in trials])
            Path(a.pick_log).write_text(json.dumps(info, indent=2))
            state["picked"] = True
            return pick
        if msg.startswith("What do you want to do"):
            if not state["saved"] and a.save:
                state["saved"] = True
                return "Save the model to a local folder"
            return ""
        raise RuntimeError(f"unexpected Heretic prompt: {msg!r}")

    def text(msg, *args, **kwargs):
        if msg.startswith("How many additional trials"):
            return str(a.more)
        raise RuntimeError(f"unexpected Heretic prompt: {msg!r}")

    def path(msg):
        if not a.save:
            raise RuntimeError("no --save path")
        return a.save

    def password(msg):
        raise RuntimeError("refusing to enter any credential")

    hm.prompt_select, hm.prompt_text, hm.prompt_path, hm.prompt_password = select, text, path, password
    hm.main()


if __name__ == "__main__":
    main()
