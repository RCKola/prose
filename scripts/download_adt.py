#!/usr/bin/env python3
"""Download ADT sequences directly from the signed-URL JSON.

The official `adt_benchmark_dataset_downloader` CLI (shipped by projectaria_tools)
is not available on Python 3.13 — so we parse the JSON manifest ourselves.

Usage:
    python scripts/download_adt.py \\
        --urls-json /path/to/ADT_download_urls.json \\
        --output /data1/nahyuk.lee/adt \\
        --sequence Apartment_release_clean_seq133_M1292

    # Download everything (warning: >2 TB total):
    python scripts/download_adt.py --urls-json ... --output ... --sequence all

    # Restrict to specific data types:
    python scripts/download_adt.py --urls-json ... --output ... \\
        --sequence Apartment_release_clean_seq133_M1292 \\
        --data-types main_vrs main_groundtruth depth
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence
from urllib.request import urlretrieve


# Reasonable default subset for the paper's pipeline:
#   main_vrs            raw RGB + calibration
#   main_groundtruth    per-frame poses + transform
#   depth               GT depth for evaluation
#   mps_slam_trajectories  alt. pose source
DEFAULT_DATA_TYPES = [
    "main_vrs",
    "main_groundtruth",
    "depth",
    "mps_slam_trajectories",
    "mps_slam_calibration",
]


def _sha1(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _progress_factory(filename: str, total_bytes: int):
    start = time.time()

    def hook(blocks: int, bs: int, total: int):
        done = min(blocks * bs, total if total > 0 else total_bytes)
        pct = 100.0 * done / max(total, total_bytes, 1)
        elapsed = time.time() - start
        speed = done / (elapsed + 1e-9) / 1e6
        bar = ("#" * int(pct / 2)).ljust(50)
        sys.stdout.write(
            f"\r  [{bar}] {pct:5.1f}%  {done / 1e6:8.1f} MB  {speed:6.1f} MB/s  {filename[:40]:<40}"
        )
        sys.stdout.flush()

    return hook


def _download_one(
    entry: dict,
    out_dir: Path,
    *,
    force: bool = False,
    unzip: bool = True,
) -> Path | None:
    filename = entry["filename"]
    url = entry["download_url"]
    expected_sha = entry.get("sha1sum")
    expected_size = int(entry.get("file_size_bytes", 0))

    target = out_dir / filename
    out_dir.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force:
        if expected_sha and _sha1(target) == expected_sha:
            print(f"  ✓ {filename} (already present, sha1 ok)")
            _maybe_unzip(target, unzip)
            return target
        print(f"  ! {filename}: exists but sha1 mismatch → re-downloading")

    print(f"  ↓ {filename}  ({expected_size / 1e6:.1f} MB)")
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        urlretrieve(url, tmp, _progress_factory(filename, expected_size))
    except Exception as e:  # noqa: BLE001
        print(f"\n  ✗ {filename}: {e}")
        return None
    print()  # newline after progress bar

    if expected_sha:
        actual_sha = _sha1(tmp)
        if actual_sha != expected_sha:
            print(f"  ✗ {filename}: sha1 mismatch (got {actual_sha}, expected {expected_sha})")
            tmp.unlink()
            return None
    tmp.rename(target)
    _maybe_unzip(target, unzip)
    return target


def _maybe_unzip(archive: Path, enabled: bool) -> None:
    if not enabled:
        return
    if archive.suffix != ".zip":
        return
    extract_dir = archive.with_suffix("")
    if extract_dir.exists():
        return
    print(f"    unzip → {extract_dir.name}/")
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        print(f"    ✗ {archive.name}: bad zip — leaving raw.")


def download_sequence(
    urls_json: dict,
    sequence_name: str,
    output_root: Path,
    data_types: Sequence[str],
    *,
    unzip: bool = True,
    force: bool = False,
) -> int:
    seq_entries = urls_json["sequences"].get(sequence_name)
    if seq_entries is None:
        print(f"✗ Sequence not found in manifest: {sequence_name}")
        return 1

    out_dir = output_root / sequence_name
    print(f"→ {sequence_name} → {out_dir}")

    requested = list(data_types) if data_types else list(seq_entries.keys())
    missing = [dt for dt in requested if dt not in seq_entries]
    if missing:
        print(f"  ! Not available in manifest: {missing}")
    requested = [dt for dt in requested if dt in seq_entries]

    total_size = sum(seq_entries[dt].get("file_size_bytes", 0) for dt in requested)
    print(f"  {len(requested)} artifacts, {total_size / 1e9:.2f} GB total")

    ok, fail = 0, 0
    for dt in requested:
        entry = seq_entries[dt]
        if not isinstance(entry, dict) or "download_url" not in entry:
            print(f"  - {dt}: no url (skipped)")
            continue
        if _download_one(entry, out_dir, force=force, unzip=unzip) is not None:
            ok += 1
        else:
            fail += 1
    print(f"  done: {ok} ok, {fail} failed")
    return 0 if fail == 0 else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urls-json", type=Path, required=True,
                    help="Path to ADT_download_urls.json.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Where to store downloads (one subdir per sequence).")
    ap.add_argument("--sequence", type=str, default="Apartment_release_clean_seq133_M1292",
                    help="Sequence name or 'all'. Default is the paper's sequence.")
    ap.add_argument("--data-types", nargs="+", default=DEFAULT_DATA_TYPES,
                    help=f"Which data types to fetch. Default: {DEFAULT_DATA_TYPES}")
    ap.add_argument("--all-data-types", action="store_true",
                    help="Shortcut: download every data type for the chosen sequence.")
    ap.add_argument("--no-unzip", action="store_true",
                    help="Keep .zip files unextracted.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if target exists with matching sha1.")
    args = ap.parse_args()

    with args.urls_json.open() as f:
        manifest = json.load(f)

    data_types = None if args.all_data_types else args.data_types
    unzip = not args.no_unzip

    if args.sequence == "all":
        seqs = sorted(manifest["sequences"].keys())
        print(f"Downloading {len(seqs)} sequences (this is large — ~2 TB). Ctrl-C to stop.")
        rc = 0
        for s in seqs:
            rc |= download_sequence(manifest, s, args.output, data_types or [],
                                    unzip=unzip, force=args.force)
        return rc

    return download_sequence(
        manifest, args.sequence, args.output, data_types or [],
        unzip=unzip, force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
