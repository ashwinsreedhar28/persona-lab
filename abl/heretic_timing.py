"""A6 timing rule (abl/PLAN.md): time Heretic's first 3 trials, then cap the trial count so the whole Heretic
run ends within 6 hours; if even a 30-trial run projects past 6 hours, Heretic is deferred to the Phase B pod.
Runs in .venv-heretic (reads Heretic's optuna journal). Writes abl/out/heretic_timing.json.

    .venv-heretic/bin/python abl/heretic_timing.py --journal abl/out/heretic/checkpoints/<model>.jsonl \
        --probe_start <epoch s> --probe_end <epoch s>

Wall clock of the full run = the probe already spent + setup again on resume (model load, refusal directions,
base evaluation) + K more trials + saving (budgeted as one more setup).
"""
import argparse
import json
import math
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

BUDGET_S = 6 * 3600
MIN_TRIALS = 30
MAX_TRIALS = 200      # Heretic's default n_trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", required=True)
    ap.add_argument("--probe_start", type=float, required=True)
    ap.add_argument("--probe_end", type=float, required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out" / "heretic_timing.json"))
    a = ap.parse_args()
    storage = JournalStorage(JournalFileBackend(a.journal, lock_obj=JournalFileOpenLock(a.journal)))
    study = optuna.load_study(study_name="heretic", storage=storage)
    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    assert len(done) >= 3, f"probe finished only {len(done)} trials"
    durs = [(t.datetime_complete - t.datetime_start).total_seconds() for t in done]
    per = sum(durs) / len(durs)
    setup = min(t.datetime_start for t in done).timestamp() - a.probe_start
    spent = a.probe_end - a.probe_start
    proj_30 = setup + MIN_TRIALS * per + setup           # a fresh 30-trial run: setup, 30 trials, save
    left = BUDGET_S - spent - 2 * setup                  # after the probe: resume setup + save
    total = min(MAX_TRIALS, len(done) + max(0, math.floor(left / per)))
    defer = proj_30 > BUDGET_S or total < MIN_TRIALS
    out = dict(rule="cap trials so the run ends within 6 h; defer if a 30-trial run projects past 6 h",
               trial_seconds=[round(d, 1) for d in durs], mean_trial_seconds=round(per, 1),
               setup_seconds=round(setup, 1), probe_seconds=round(spent, 1),
               projected_30_trial_hours=round(proj_30 / 3600, 2), decision="defer" if defer else "run",
               total_trials=None if defer else total, more_trials=None if defer else total - len(done),
               projected_total_hours=None if defer else round((spent + 2 * setup + (total - len(done)) * per) / 3600, 2),
               probe_refusals=[t.user_attrs.get("refusals") for t in done],
               probe_kl=[round(t.user_attrs.get("kl_divergence", float("nan")), 4) for t in done])
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
