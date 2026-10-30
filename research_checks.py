"""Small, model-free validation and provenance helpers for research runs."""
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np


def unit_direction(vector, *, require_unit=False):
    """Validate a single direction; normalize in float64 before returning fp32.

    Projection callers must require a unit input: silently accepting a scaled
    r-hat would turn I-rr^T into a non-idempotent operation.
    """
    v = np.asarray(vector, dtype=np.float64)
    if v.ndim != 1 or not v.size or not np.isfinite(v).all():
        raise ValueError("direction must be a finite, nonempty vector")
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("direction must have finite nonzero norm")
    if require_unit and not np.isclose(norm, 1, atol=1e-4, rtol=0):
        raise ValueError("projection requires a unit direction")
    return (v if require_unit else v / norm).astype(np.float32)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def provenance(paths, settings):
    """Record content hashes, installed versions and Git state; never prompt text."""
    root = Path(__file__).resolve().parent
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    versions = {}
    for name in ("numpy", "mlx", "mlx-lm", "transformers", "huggingface-hub"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(files={str(p): file_hash(p) for p in paths}, settings=settings,
                python=platform.python_version(), platform=platform.platform(),
                versions=versions, git_head=git("rev-parse", "HEAD"),
                git_status=git("status", "--porcelain"))


def normalized_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def word_grams(text, n=8):
    words = normalized_text(text).split()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def split_overlap(splits):
    """Report exact and shared 8-gram overlap using indices only; do not resplit.

    Shared grams flag possible contamination, including generic boilerplate;
    they are not a semantic similarity test.
    """
    normalized = {k: [normalized_text(s) for s in rows] for k, rows in splits.items()}
    grams = {k: [word_grams(s) for s in rows] for k, rows in splits.items()}
    duplicates = {k: len(rows) - len(set(rows)) for k, rows in normalized.items()}
    pairs = []
    for a, b in combinations(splits, 2):
        lookup, gram_lookup = {}, {}
        for j, text in enumerate(normalized[b]):
            lookup.setdefault(text, []).append(j)
            for gram in grams[b][j]:
                gram_lookup.setdefault(gram, set()).add(j)
        exact, shared = [], []
        for i, text in enumerate(normalized[a]):
            matches = lookup.get(text, [])
            exact.extend([i, j] for j in matches)
            hits = set().union(*(gram_lookup.get(g, set()) for g in grams[a][i]))
            shared.extend([i, j] for j in sorted(hits - set(matches)))
        pairs.append(dict(a=a, b=b, exact=exact, shared_8gram=shared))
    return dict(sizes={k: len(v) for k, v in splits.items()}, duplicates=duplicates, pairs=pairs)
