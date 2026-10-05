"""Atomic JSON writes, versioned CSV exports, and source hashes for experiment provenance."""

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
    """Atomically replace JSON, optionally keeping one rolling .bak copy."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)

    if backup and os.path.exists(path):
        try:
            shutil.copy2(path, path + ".bak")
        except OSError:
            pass  # a failed backup must not block the write

    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)  # atomic on POSIX
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
    """Summarize timestamped versions of a CSV file."""
    stem, ext = os.path.splitext(path)
    base = os.path.basename(stem)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    hits = sorted(
        f
        for f in os.listdir(directory)
        if f.startswith(base + "__") and f.endswith(ext)
    )
    if not hits:
        return f"{base}: no previous versions"
    return f"{base}: {len(hits)} archived version(s), earliest {hits[0]}, latest {hits[-1]}"
