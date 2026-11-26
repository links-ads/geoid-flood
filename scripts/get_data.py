#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import sys
import tarfile
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download, hf_hub_url
from huggingface_hub.utils import build_hf_headers, disable_progress_bars, get_session
from tqdm import tqdm

try:
    from dotenv import load_dotenv
except ImportError:
    # The downloader also works with variables set in the calling shell.  Do
    # not make an optional .env convenience a hard runtime dependency.
    pass
else:
    load_dotenv()

REPO = "links-ads/geoid-flood"
REVISION = "v1.1"  # dataset tag this code release reads; bump with each dataset release
TREES = ("geoid-flood", "geoid-flood-heldout")
LAYERS = ("s1grd", "s1rtc", "s2l2a", "dem", "label", "cloudmask", "floodmask", "permwater", "validity")
SPLITS = ("train", "val", "test")
METADATA = ("data_tiles_s256_st128.csv", "tile_catalog.parquet")
STAGING = ".geoid_flood_staging"
STATE = ".geoid_flood_done"
STATE_REVISION = ".geoid_flood_revision"

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

SHARD_RE = re.compile(r"^(?P<tree>[^/]+)/shards/(?P<split>[^/]+)/(?P<layer>[^/]+)/[^/]+\.tar$")


def _safe_dest(root: Path, name: str) -> Path:
    root = root.resolve()
    dest = (root / name).resolve()
    try:
        dest.relative_to(root)
    except ValueError:
        raise ValueError(f"member escapes destination: {name}")
    return dest


def _safe_members(tar, root: Path):
    for info in tar:
        if not info.isreg():
            raise ValueError(f"unexpected member type in shard: {info.name}")
        yield info, _safe_dest(root, info.name)


class TransientRangeError(Exception):
    pass


def select_shards(index: dict, trees, splits, layers) -> list[dict]:
    out = []
    for shard in index["shards"]:
        m = SHARD_RE.match(shard["path"])
        if m and m["tree"] in trees and m["split"] in splits and m["layer"] in layers:
            out.append(shard)
    return out


def select_members(index: dict, trees, splits, layers, tiles) -> list[dict]:
    out, found = [], set()
    for shard in select_shards(index, trees, splits, layers):
        if "member_index" not in shard:
            sys.exit("this dataset revision has no per-file index, so --tile is unavailable; "
                     "use --split/--layer to download whole shards")
        if len(shard["members"]) != len(shard["member_index"]):
            sys.exit(f"{shard['path']}: the index lists {len(shard['members'])} files "
                     f"but {len(shard['member_index'])} byte ranges")
        tree = SHARD_RE.match(shard["path"])["tree"]
        for rel, (offset, size, sha256) in zip(shard["members"], shard["member_index"]):
            name = rel.rsplit("/", 1)[-1]
            tile = next((t for t in tiles if name.startswith(f"{t}_")), None)
            if tile:
                found.add(tile)
                out.append({"shard": shard["path"], "tree": tree, "rel": rel,
                            "offset": offset, "size": size, "sha256": sha256})
    if found != set(tiles):
        sys.exit(f"no files found for tile(s) {sorted(set(tiles) - found)} with the selected --tree/--split/--layer")
    return out


def fetch_range(repo: str, path: str, revision: str, offset: int, size: int, token: str | None) -> bytes:
    last = offset + size - 1
    headers = {**build_hf_headers(token=token), "Range": f"bytes={offset}-{last}", "Accept-Encoding": "identity"}
    url = hf_hub_url(repo, path, repo_type="dataset", revision=revision)
    chunks, received = [], 0
    with get_session().stream("GET", url, headers=headers, follow_redirects=True) as response:
        if response.status_code in RETRYABLE_STATUS:
            raise TransientRangeError(f"range request for {path} returned {response.status_code}")
        content_range = response.headers.get("Content-Range", "")
        if response.status_code != 206 or not content_range.startswith(f"bytes {offset}-{last}/"):
            raise ValueError(f"range request for {path} returned {response.status_code} "
                             f"{content_range!r}, expected bytes {offset}-{last}")
        for chunk in response.iter_bytes():
            chunks.append(chunk)
            received += len(chunk)
            if received > size:
                break
    if received != size:
        raise ValueError(f"range request for {path} returned {received} of {size} bytes")
    return b"".join(chunks)


def with_retries(call, attempts: int = 3):
    for attempt in range(attempts):
        try:
            return call()
        except (httpx.TransportError, TransientRangeError):
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_member(repo: str, member: dict, revision: str, dest: Path, token: str | None) -> None:
    target = _safe_dest(dest / member["tree"], member["rel"])
    if target.is_file() and target.stat().st_size == member["size"]:
        return
    data = with_retries(lambda: fetch_range(repo, member["shard"], revision, member["offset"], member["size"], token))
    if hashlib.sha256(data).hexdigest() != member["sha256"]:
        raise ValueError(f"sha256 mismatch for {member['rel']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(target)


def download_members(args, index: dict) -> None:
    members = select_members(index, args.tree, args.split, args.layer, args.tile)
    total = sum(m["size"] for m in members)
    print(f"GEOID-Flood {args.revision}: {len(members)} files from {len({m['shard'] for m in members})} "
          f"shards, {total/1e6:.1f} MB")
    if args.list:
        for m in members:
            print(f"  {m['tree']}/{m['rel']:70} {m['size']:>10,} B")
        return
    args.dest.mkdir(parents=True, exist_ok=True)
    check_revision(args.dest, args.revision)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_member, args.repo, m, args.revision, args.dest, args.token): m
                   for m in members}
        try:
            for future, member in futures.items():
                try:
                    future.result()
                except Exception as exc:
                    failures.append((member["rel"], f"{type(exc).__name__}: {exc}"))
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    if failures:
        print(f"\n{len(failures)} file(s) failed; re-run to retry them:", file=sys.stderr)
        for rel, err in failures:
            print(f"  {rel}: {err}", file=sys.stderr)
        sys.exit(1)
    print(f"done -> {args.dest}/")


def fetch_shard(repo: str, path: str, revision: str, staging: Path, token: str | None) -> Path:
    """Pull one shard to staging over the Xet chunk protocol."""
    return Path(hf_hub_download(repo_id=repo, filename=path, repo_type="dataset", revision=revision,
                                local_dir=str(staging), token=token))


def check_revision(dest: Path, revision: str) -> None:
    """Refuse to resume a download of another dataset revision: shard names repeat across versions."""
    marker = dest / STATE_REVISION
    if marker.exists():
        recorded = marker.read_text().strip()
    elif (dest / STATE).exists():
        recorded = "v1.0"  # state files written before revisions were recorded are all v1.0
    else:
        marker.write_text(revision + "\n")
        return
    if recorded != revision:
        sys.exit(f"{dest} holds GEOID-Flood {recorded}, not {revision}. "
                 f"Download into a new --dest, or pass --revision {recorded} to resume.")
    marker.write_text(revision + "\n")


def verify_shard(tar_path: Path, expect_sha: str | None) -> None:
    if not expect_sha:
        return
    h = hashlib.sha256()
    with open(tar_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != expect_sha:
        raise ValueError(f"sha256 mismatch for {tar_path.name}")


def unpack_shard(tar_path: Path, out_root: Path) -> None:
    """Extract a staged shard into out_root. Writes via .part so a kill leaves no half files."""
    with tarfile.open(tar_path, mode="r") as tar:
        for info, dest in _safe_members(tar, out_root):
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tar.extractfile(info) as src, open(tmp, "wb") as fh:
                shutil.copyfileobj(src, fh, 1 << 20)
            tmp.replace(dest)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download GEOID-Flood from the Hugging Face Hub and unpack it to the expected layout"
    )
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default=REVISION,
                    help=f"dataset version tag, branch or commit (default {REVISION})")
    ap.add_argument("--dest", default="data", type=Path, help="parent dir; trees are created under it")
    ap.add_argument("--tree", nargs="+", choices=TREES, default=list(TREES))
    ap.add_argument("--split", nargs="+", choices=SPLITS, default=list(SPLITS))
    ap.add_argument("--layer", nargs="+", choices=LAYERS, default=list(LAYERS))
    ap.add_argument("--tile", nargs="+", metavar="TILE_ID",
                    help="download only these tiles (e.g. EMSR720-4-1) by reading their byte ranges "
                         "out of the shards, instead of whole shards. The metadata CSV/parquet "
                         "are not fetched")
    ap.add_argument("--workers", type=int, default=4,
                    help="shards downloaded concurrently (default 4); raises peak disk use. "
                         "With --tile: files fetched concurrently, no staging disk")
    ap.add_argument("--extract-workers", type=int, default=None,
                    help="shards extracted concurrently (default: same as --workers)")
    ap.add_argument("--force", action="store_true", help="skip the free-space preflight check")
    ap.add_argument("--list", action="store_true", help="print the selection and exit")
    ap.add_argument("--sample", action="store_true",
                    help="fetch only the two-AoI EMSR712 sample: 47 tiles, all nine layers, "
                         "~2.9 GB, and exit")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    if args.sample and args.tile:
        ap.error("--sample and --tile cannot be combined")
    if args.workers < 1:
        ap.error("--workers must be at least 1")
    if args.extract_workers is None:
        args.extract_workers = args.workers
    elif args.extract_workers < 1:
        ap.error("--extract-workers must be at least 1")

    if args.sample:
        from huggingface_hub import snapshot_download

        tmp = snapshot_download(
            repo_id=args.repo, repo_type="dataset", revision=args.revision,
            allow_patterns=["sample/*"], token=args.token,
        )
        root = Path(tmp) / "sample"
        for src in root.rglob("*"):
            if src.is_file():
                dst = args.dest / src.relative_to(root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), dst)
        print(f"sample -> {args.dest}/geoid-flood/")
        print("run a config against it with:")
        print("  --data.init_args.metadata_filename data_tiles_s256_st128_sample.csv")
        return

    index_path = hf_hub_download(
        repo_id=args.repo, filename="shard_index.json.gz", repo_type="dataset",
        revision=args.revision, token=args.token,
    )
    with gzip.open(index_path, "rt") as fh:
        index = json.load(fh)

    if args.tile:
        download_members(args, index)
        return

    selected = select_shards(index, args.tree, args.split, args.layer)
    total = sum(s["size"] for s in selected)
    print(f"GEOID-Flood {args.revision}: {len(selected)} shards, {total/1e9:.1f} GB")
    if args.list:
        by = {}
        for s in selected:
            m = SHARD_RE.match(s["path"])
            k = (m["tree"], m["split"], m["layer"])
            a, b = by.get(k, (0, 0))
            by[k] = (a + 1, b + s["size"])
        for k in sorted(by):
            n, b = by[k]
            print(f"  {'/'.join(k):45} {n:4d} shards  {b/1e9:8.2f} GB")
        return

    args.dest.mkdir(parents=True, exist_ok=True)
    check_revision(args.dest, args.revision)
    state = args.dest / STATE
    done = set(state.read_text().split()) if state.exists() else set()
    todo = [s for s in selected if s["path"] not in done]
    remaining = sum(s["size"] for s in todo)

    if todo:
        staging_slots = args.workers + args.extract_workers
        need = remaining + staging_slots * max(s["size"] for s in todo)
        free = shutil.disk_usage(args.dest).free
        if free < need and not args.force:
            sys.exit(f"not enough space at {args.dest}: {free/1e9:.1f} GB free, "
                     f"need ~{need/1e9:.1f} GB ({remaining/1e9:.1f} GB of data plus staging). "
                     f"Narrow the selection with --layer/--split/--tree, or pass --force.")

    for tree in args.tree:
        for name in METADATA:
            p = hf_hub_download(
                repo_id=args.repo, filename=f"{tree}/{name}", repo_type="dataset",
                revision=args.revision, local_dir=args.dest, token=args.token,
            )
            print(f"metadata {p}")

    staging = args.dest / STAGING
    shutil.rmtree(staging, ignore_errors=True)  # a killed run leaves tars behind
    staging.mkdir(parents=True)


    disable_progress_bars()
    lock = threading.Lock()
    failures: list[tuple[str, str]] = []
    n_done = len(selected) - len(todo)
    done_bytes = 0
    stop = threading.Event()

    def staged_bytes() -> int:
        n = 0
        for p in staging.rglob("*"):  # the in-flight tars plus the .incomplete parts under them
            if p.is_file():
                try:
                    n += p.stat().st_size
                except OSError:  # unlinked between walk and stat
                    pass
        return n

    def monitor() -> None:
        reported = 0
        while not stop.is_set():
            with lock:
                target = done_bytes + staged_bytes()
                if target > reported:
                    bar.update(target - reported)
                    reported = target
            stop.wait(0.5)

    def download_shard(shard: dict) -> Path:
        """Download and checksum a shard before it is eligible for extraction."""
        tar = fetch_shard(args.repo, shard["path"], args.revision, staging, args.token)
        try:
            verify_shard(tar, shard.get("sha256"))
        except BaseException:
            tar.unlink(missing_ok=True)
            raise
        return tar

    def extract_shard(shard: dict, tar: Path) -> None:
        """Extract only a verified shard; individual files remain atomic via .part."""
        unpack_shard(tar, args.dest / shard["path"].split("/", 1)[0])

    def mark_complete(shard: dict, tar: Path) -> None:
        nonlocal n_done, done_bytes
        with lock:
            done_bytes += shard["size"]
            tar.unlink(missing_ok=True)
            fh.write(shard["path"] + "\n")
            fh.flush()
            n_done += 1
            bar.set_description_str(f"{n_done}/{len(selected)} shards", refresh=False)

    with state.open("a") as fh, \
            tqdm(total=total, initial=total - remaining, unit="B", unit_scale=True,
                 unit_divisor=1000, smoothing=0.05,
                 desc=f"{n_done}/{len(selected)} shards") as bar, \
            ThreadPoolExecutor(max_workers=args.workers) as download_pool, \
            ThreadPoolExecutor(max_workers=args.extract_workers) as extract_pool:
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        try:
            # Keep a bounded pipeline: at most --workers tars are downloading
            # and at most --extract-workers verified tars are being unpacked.
            # This leaves the network active while earlier shards are written.
            pending_downloads: dict[Future, dict] = {}
            pending_extractions: dict[Future, tuple[dict, Path]] = {}
            ready_to_extract: deque[tuple[dict, Path]] = deque()
            retry_queue: deque[dict] = deque()
            shard_iter = iter(todo)
            attempts: dict[str, int] = {}

            def submit_download(shard: dict) -> None:
                attempts[shard["path"]] = attempts.get(shard["path"], 0) + 1
                pending_downloads[download_pool.submit(download_shard, shard)] = shard

            def retry_or_record(shard: dict, exc: Exception) -> None:
                if attempts[shard["path"]] < 2:
                    retry_queue.append(shard)
                else:
                    failures.append((shard["path"], f"{type(exc).__name__}: {exc}"))

            def fill_download_slots() -> bool:
                # Tars live in staging until extraction finishes, so cap the
                # whole pipeline at what the free-space preflight reserved.
                slots = args.workers + args.extract_workers
                submitted = False
                while (len(pending_downloads) < args.workers
                       and len(pending_downloads) + len(ready_to_extract)
                       + len(pending_extractions) < slots):
                    if retry_queue:
                        shard = retry_queue.popleft()
                    else:
                        try:
                            shard = next(shard_iter)
                        except StopIteration:
                            break
                    submit_download(shard)
                    submitted = True
                return submitted

            fill_download_slots()
            while pending_downloads or pending_extractions or ready_to_extract or retry_queue:
                progressed = False

                for future, shard in list(pending_downloads.items()):
                    if not future.done():
                        continue
                    del pending_downloads[future]
                    try:
                        ready_to_extract.append((shard, future.result()))
                    except Exception as exc:
                        retry_or_record(shard, exc)
                    progressed = True

                for future, (shard, tar) in list(pending_extractions.items()):
                    if not future.done():
                        continue
                    del pending_extractions[future]
                    try:
                        future.result()
                    except Exception as exc:
                        tar.unlink(missing_ok=True)
                        retry_or_record(shard, exc)
                    else:
                        mark_complete(shard, tar)
                    progressed = True

                while ready_to_extract and len(pending_extractions) < args.extract_workers:
                    shard, tar = ready_to_extract.popleft()
                    future = extract_pool.submit(extract_shard, shard, tar)
                    pending_extractions[future] = (shard, tar)
                    progressed = True

                if fill_download_slots():
                    progressed = True
                if progressed:
                    continue

                active = set(pending_downloads) | set(pending_extractions)
                if active:
                    wait(active, return_when=FIRST_COMPLETED)
        finally:
            stop.set()
            watcher.join()

    shutil.rmtree(staging, ignore_errors=True)

    if failures:
        print(f"\n{len(failures)} shard(s) failed; re-run to retry them:", file=sys.stderr)
        for path, err in failures:
            print(f"  {path}: {err}", file=sys.stderr)
        sys.exit(1)

    print(f"\ndone -> {args.dest}/")
    expected = Path("data").resolve()
    if args.dest.resolve() != expected:
        print("configs read data/geoid-flood; link the trees with:")
        for tree in args.tree:
            print(f"  ln -s {args.dest.resolve()}/{tree} data/{tree}")


if __name__ == "__main__":
    main()
