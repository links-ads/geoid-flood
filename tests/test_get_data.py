"""Pipeline invariants for scripts/get_data.py, with the Hub stubbed out."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tarfile
import threading
import time
import types
from pathlib import Path

import pytest

SHARD_COUNT = 12
SHARD_PAYLOAD = 64_000
EXTRACT_DELAY = 0.05  # makes extraction the bottleneck, so the queue would grow


def _stub_hub() -> None:
    """get_data imports huggingface_hub at module scope; tests never hit the network."""
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda **kw: None
    hub.snapshot_download = lambda **kw: None
    utils = types.ModuleType("huggingface_hub.utils")
    utils.disable_progress_bars = lambda *a, **k: None
    hub.utils = utils
    sys.modules.setdefault("huggingface_hub", hub)
    sys.modules.setdefault("huggingface_hub.utils", utils)


@pytest.fixture(scope="module")
def get_data():
    _stub_hub()
    path = Path(__file__).resolve().parents[1] / "scripts" / "get_data.py"
    spec = importlib.util.spec_from_file_location("get_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hub(tmp_path, get_data):
    """A fake Hub tree of tar shards plus the gzipped index get_data expects."""
    root = tmp_path / "hub"
    shards = []
    for i in range(SHARD_COUNT):
        rel = f"geoid-flood/shards/train/s1grd/shard-{i:04d}.tar"
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dest, "w") as tar:
            info = tarfile.TarInfo(f"train/s1grd/tile-{i:04d}.tif")
            info.size = SHARD_PAYLOAD
            tar.addfile(info, io.BytesIO(bytes(SHARD_PAYLOAD)))
        raw = dest.read_bytes()
        shards.append({"path": rel, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})

    index = root / "shard_index.json.gz"
    with gzip.open(index, "wt") as fh:
        json.dump({"shards": shards}, fh)
    for tree in get_data.TREES:
        for name in get_data.METADATA:
            meta = root / tree / name
            meta.parent.mkdir(parents=True, exist_ok=True)
            meta.write_bytes(b"x")

    def download(filename=None, local_dir=None, **kw):
        if filename == "shard_index.json.gz":
            return str(index)
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / filename, out)
        return str(out)

    return download


def _run(get_data, monkeypatch, hub, dest, argv):
    """Run main() against the fake Hub; return the peak tar count seen in staging."""
    monkeypatch.setattr(get_data, "hf_hub_download", hub)
    monkeypatch.setattr(get_data, "disable_progress_bars", lambda *a, **k: None)

    real_unpack = get_data.unpack_shard

    def slow_unpack(tar_path, out_root):
        time.sleep(EXTRACT_DELAY)
        real_unpack(tar_path, out_root)

    monkeypatch.setattr(get_data, "unpack_shard", slow_unpack)
    monkeypatch.setattr(sys, "argv", ["get_data.py", "--dest", str(dest), "--force", *argv])

    staging = dest / get_data.STAGING
    peak = 0
    stop = threading.Event()

    def sample():
        nonlocal peak
        while not stop.is_set():
            try:
                peak = max(peak, sum(1 for p in staging.rglob("*.tar") if p.is_file()))
            except OSError:
                pass
            stop.wait(0.005)

    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    try:
        get_data.main()
    finally:
        stop.set()
        watcher.join()
    return peak


def test_staging_is_bounded_when_extraction_lags(get_data, monkeypatch, hub, tmp_path):
    """Downloads must not queue up unbounded tars: that is what the preflight reserves."""
    dest = tmp_path / "data"
    workers, extract_workers = 4, 1
    peak = _run(get_data, monkeypatch, hub, dest,
                ["--workers", str(workers), "--extract-workers", str(extract_workers)])
    assert peak <= workers + extract_workers, (
        f"{peak} tars staged at once, preflight only reserves {workers + extract_workers}"
    )


def test_all_shards_extracted_and_recorded(get_data, monkeypatch, hub, tmp_path):
    dest = tmp_path / "data"
    _run(get_data, monkeypatch, hub, dest, ["--workers", "2"])
    assert len(list((dest / "geoid-flood" / "train" / "s1grd").glob("*.tif"))) == SHARD_COUNT
    assert len((dest / get_data.STATE).read_text().split()) == SHARD_COUNT
    assert not (dest / get_data.STAGING).exists()


def test_extract_workers_defaults_to_workers(get_data, monkeypatch, hub, tmp_path, capsys):
    """A --workers bump should scale extraction too, not leave it pinned low."""
    seen = {}
    real_pool = get_data.ThreadPoolExecutor

    def spy(max_workers=None, **kw):
        seen.setdefault("pools", []).append(max_workers)
        return real_pool(max_workers=max_workers, **kw)

    monkeypatch.setattr(get_data, "ThreadPoolExecutor", spy)
    _run(get_data, monkeypatch, hub, tmp_path / "data", ["--workers", "3"])
    assert seen["pools"] == [3, 3]


def test_member_escaping_destination_is_rejected(get_data, tmp_path):
    """The traversal guard must survive the Windows-separator fix."""
    root = tmp_path / "out"
    root.mkdir()
    evil = tarfile.TarInfo("../escaped.tif")
    evil.size = 0
    with pytest.raises(ValueError, match="escapes destination"):
        list(get_data._safe_members([evil], root))


def test_sibling_prefix_is_not_mistaken_for_a_child(get_data, tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out-evil").mkdir()
    member = tarfile.TarInfo("../out-evil/x.tif")
    member.size = 0
    with pytest.raises(ValueError, match="escapes destination"):
        list(get_data._safe_members([member], tmp_path / "out"))
