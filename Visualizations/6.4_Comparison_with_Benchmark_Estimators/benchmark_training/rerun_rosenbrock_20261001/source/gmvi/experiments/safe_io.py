"""
Non-destructive result writing.

Three problems this fixes, in order of how much they can cost you.

1. NON-ATOMIC CACHE WRITES.  The sweep scripts do

       json.dump(out, open(path, "w"))

   after every completed job. `open(path, "w")` truncates the file first, so an
   interrupt, a crash or a full disk between truncation and the end of the dump
   leaves a corrupt JSON file -- and with it the entire shard's completed runs,
   which at n = 100 is hours of compute. `atomic_json_dump` writes to a
   temporary file in the same directory and then `os.replace`s it, which is
   atomic on POSIX: the destination is either the old file or the new one,
   never a half-written one.

2. OVERWRITTEN RESULT FILES.  `df.to_csv(f"{BASE}.csv")` replaces the previous
   run's output. In practice the CSV is regenerated from the accumulating
   cache, so a rerun usually produces a SUPERSET and nothing is lost -- but
   that is a property of the cache, not a guarantee. If the cache is ever
   cleared, an arm is renamed, or the script is run with a different --mode,
   the regenerated CSV is a subset and the previous one is gone.
   `save_versioned` writes a timestamped copy that is never touched again, and
   updates the stable `{BASE}.csv` name that notebooks and the thesis refer to.

3. NO PROVENANCE.  Nothing in a result file records which version of which
   script produced it, so a script edited after a run cannot be matched to its
   output except by comparing timestamps. `source_hash` returns a short digest
   of the calling file; put it in a column and the question answers itself.

Nothing here deletes anything. Old timestamped files accumulate; they are a few
hundred KB each and are the cheapest insurance in the project.
"""

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Optional


def source_hash(path: str, length: int = 12) -> str:
    """Short digest of a source file, for a provenance column."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:length]


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_json_dump(obj, path: str, backup: bool = True) -> None:
    """
    Write JSON so that `path` is never left in a partially written state.

    Keeps one rolling backup at `path + ".bak"`, so even a logically wrong
    write (an empty cache, say) is recoverable.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)

    if backup and os.path.exists(path):
        try:
            shutil.copy2(path, path + ".bak")
        except OSError:
            pass                      # a failed backup must not block the write

    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)         # atomic on POSIX
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def save_versioned(df, path: str, also_stable: bool = True) -> str:
    """
    Write `df` to a timestamped file that will never be overwritten, and
    optionally refresh the stable name too.

        save_versioned(df, ".../sigma_collapse_arms.csv")
          -> sigma_collapse_arms__20260828T233612Z.csv   (permanent)
          -> sigma_collapse_arms.csv                     (latest, for notebooks)

    Returns the path of the timestamped file.
    """
    stem, ext = os.path.splitext(path)
    versioned = f"{stem}__{timestamp()}{ext}"
    df.to_csv(versioned, index=False)
    if also_stable:
        shutil.copy2(versioned, path)
    return versioned


def describe_history(path: str) -> str:
    """One-line summary of the versioned files sitting next to `path`."""
    stem, ext = os.path.splitext(path)
    base = os.path.basename(stem)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    hits = sorted(f for f in os.listdir(directory)
                  if f.startswith(base + "__") and f.endswith(ext))
    if not hits:
        return f"{base}: no previous versions"
    return f"{base}: {len(hits)} archived version(s), earliest {hits[0]}, latest {hits[-1]}"
