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
    hub.hf_hub_url = lambda *a, **kw: None
    utils = types.ModuleType("huggingface_hub.utils")
    utils.disable_progress_bars = lambda *a, **k: None
    utils.build_hf_headers = lambda **kw: {}
    utils.get_session = lambda: None
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


def test_every_hub_call_pins_the_revision(get_data, monkeypatch, hub, tmp_path):
    seen = []

    def spy(**kw):
        seen.append(kw.get("revision"))
        return hub(**kw)

    _run(get_data, monkeypatch, spy, tmp_path / "data", ["--revision", "v9.9"])
    assert seen and set(seen) == {"v9.9"}


def test_resuming_another_revision_is_refused(get_data, monkeypatch, hub, tmp_path):
    dest = tmp_path / "data"
    _run(get_data, monkeypatch, hub, dest, ["--revision", "v1.0"])
    with pytest.raises(SystemExit, match="holds GEOID-Flood v1.0, not v1.1"):
        _run(get_data, monkeypatch, hub, dest, ["--revision", "v1.1"])


def test_state_without_revision_is_read_as_v1_0(get_data, tmp_path):
    """Downloads made before revisions were recorded all came from v1.0."""
    (tmp_path / get_data.STATE).write_text("some/shard.tar\n")
    get_data.check_revision(tmp_path, "v1.0")
    with pytest.raises(SystemExit):
        get_data.check_revision(tmp_path, "v1.1")


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


TILE_A = "EMSR111-1-0"
TILE_B = "EMSR222-2-5"


@pytest.fixture
def tile_hub(tmp_path, get_data, monkeypatch):
    root = tmp_path / "hub"
    rel_dir = "geoid-flood/shards/train/s1rtc"
    shard_path = f"{rel_dir}/train-s1rtc-00000-of-00001.tar"
    shard_file = root / shard_path
    shard_file.parent.mkdir(parents=True)
    names = [f"EMSR111-1/s1rtc/{TILE_A}_s1rtc_pre_20200101T000000.tif",
             f"EMSR111-1/s1rtc/{TILE_A}_s1rtc_post_20200201T000000.tif",
             f"EMSR222-2/s1rtc/{TILE_B}_s1rtc_pre_20200101T000000.tif"]
    payloads = {n: bytes([i + 1]) * (300 + 100 * i) for i, n in enumerate(names)}
    with tarfile.open(shard_file, "w", format=tarfile.PAX_FORMAT) as tar:
        for name, data in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with tarfile.open(shard_file) as tar:
        located = {i.name: (i.offset_data, i.size) for i in tar}
    shard = {"path": shard_path, "size": shard_file.stat().st_size,
             "sha256": hashlib.sha256(shard_file.read_bytes()).hexdigest(),
             "n_members": len(names), "members": names,
             "member_index": [[*located[n], hashlib.sha256(payloads[n]).hexdigest()] for n in names]}
    index = root / "shard_index.json.gz"
    with gzip.open(index, "wt") as fh:
        json.dump({"shards": [shard]}, fh)

    ranges = []

    def fetch_range(repo, path, revision, offset, size, token):
        ranges.append((offset, size))
        return shard_file.read_bytes()[offset:offset + size]

    def download(filename=None, **kw):
        assert filename == "shard_index.json.gz", "tile mode must not fetch whole shards or metadata"
        return str(index)

    monkeypatch.setattr(get_data, "hf_hub_download", download)
    monkeypatch.setattr(get_data, "fetch_range", fetch_range)
    return payloads, ranges


def _run_tiles(get_data, monkeypatch, dest, argv):
    monkeypatch.setattr(sys, "argv", ["get_data.py", "--dest", str(dest), *argv])
    get_data.main()


def test_tile_mode_downloads_only_requested_files(get_data, monkeypatch, tile_hub, tmp_path):
    payloads, ranges = tile_hub
    dest = tmp_path / "data"
    _run_tiles(get_data, monkeypatch, dest, ["--tile", TILE_A])
    written = {p.relative_to(dest / "geoid-flood").as_posix(): p.read_bytes()
               for p in (dest / "geoid-flood").rglob("*.tif")}
    assert written == {n: d for n, d in payloads.items() if f"/{TILE_A}_" in n}
    assert len(ranges) == 2


def test_tile_mode_rejects_corrupt_ranges(get_data, monkeypatch, tile_hub, tmp_path):
    monkeypatch.setattr(get_data, "fetch_range", lambda *a, **k: b"\0" * a[4])
    dest = tmp_path / "data"
    with pytest.raises(SystemExit):
        _run_tiles(get_data, monkeypatch, dest, ["--tile", TILE_B])
    assert not list(dest.rglob("*.tif")) and not list(dest.rglob("*.part"))


def test_tile_mode_resumes_without_refetching(get_data, monkeypatch, tile_hub, tmp_path):
    _, ranges = tile_hub
    dest = tmp_path / "data"
    _run_tiles(get_data, monkeypatch, dest, ["--tile", TILE_A, TILE_B])
    fetched = len(ranges)
    _run_tiles(get_data, monkeypatch, dest, ["--tile", TILE_A, TILE_B])
    assert len(ranges) == fetched


def test_unknown_tile_is_an_error(get_data, monkeypatch, tile_hub, tmp_path):
    with pytest.raises(SystemExit, match="EMSR999-9-9"):
        _run_tiles(get_data, monkeypatch, tmp_path / "data", ["--tile", TILE_A, "EMSR999-9-9"])


def test_tile_mode_honours_layer_filter(get_data, monkeypatch, tile_hub, tmp_path):
    with pytest.raises(SystemExit, match=TILE_A):
        _run_tiles(get_data, monkeypatch, tmp_path / "data", ["--tile", TILE_A, "--layer", "label"])


def test_tile_mode_needs_a_member_index(get_data, monkeypatch, hub, tmp_path):
    monkeypatch.setattr(get_data, "hf_hub_download", hub)
    with pytest.raises(SystemExit, match="no per-file index"):
        _run_tiles(get_data, monkeypatch, tmp_path / "data", ["--tile", TILE_A, "--layer", "s1grd"])


class _Response:
    def __init__(self, status, content_range, body):
        self.status_code = status
        self.headers = {"Content-Range": content_range} if content_range else {}
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_bytes(self):
        half = len(self._body) // 2
        yield self._body[:half]
        yield self._body[half:]


class _Session:
    def __init__(self, response):
        self.response = response
        self.sent = []

    def stream(self, method, url, headers=None, follow_redirects=False):
        self.sent.append(headers)
        return self.response


def _range(get_data, monkeypatch, response, offset=100, size=10):
    session = _Session(response)
    monkeypatch.setattr(get_data, "get_session", lambda: session)
    monkeypatch.setattr(get_data, "hf_hub_url", lambda *a, **k: "https://hub/x")
    result = get_data.fetch_range("r", "p.tar", "rev", offset, size, None)
    return result, session


def test_fetch_range_sends_an_inclusive_range_and_returns_the_bytes(get_data, monkeypatch):
    body = bytes(range(10))
    result, session = _range(get_data, monkeypatch, _Response(206, "bytes 100-109/5000", body))
    assert result == body
    assert session.sent[0]["Range"] == "bytes=100-109"
    assert session.sent[0]["Accept-Encoding"] == "identity"


@pytest.mark.parametrize("status, content_range, body", [
    (200, None, bytes(10)),
    (206, "bytes 0-9/5000", bytes(10)),
    (206, "bytes 100-105/106", bytes(6)),
    (206, "bytes 100-109/5000", bytes(9)),
    (206, "bytes 100-109/5000", bytes(11)),
])
def test_fetch_range_rejects_anything_but_the_exact_range(get_data, monkeypatch, status, content_range, body):
    with pytest.raises(ValueError):
        _range(get_data, monkeypatch, _Response(status, content_range, body))


def test_fetch_range_stops_reading_an_oversized_body(get_data, monkeypatch):
    class Endless(_Response):
        def iter_bytes(self):
            for _ in range(1000):
                self.read = getattr(self, "read", 0) + 1
                yield bytes(10)

    response = Endless(206, "bytes 100-109/5000", b"")
    with pytest.raises(ValueError):
        _range(get_data, monkeypatch, response)
    assert response.read <= 2


def _index(*names, indexed=None):
    members = list(names)
    ranges = [[i, 1, "0" * 64] for i, _ in enumerate(members)] if indexed is None else indexed
    return {"shards": [{"path": "geoid-flood/shards/train/s1rtc/train-s1rtc-00000-of-00001.tar",
                        "members": members, "member_index": ranges}]}


def _select(get_data, index, *tiles):
    return get_data.select_members(index, ("geoid-flood",), ("train",), ("s1rtc",), list(tiles))


def test_tile_id_is_not_a_prefix_of_a_longer_tile(get_data):
    index = _index("EMSR1-1/s1rtc/EMSR1-1-1_s1rtc_pre_a.tif", "EMSR1-1/s1rtc/EMSR1-1-10_s1rtc_pre_a.tif")
    picked = _select(get_data, index, "EMSR1-1-1")
    assert [m["rel"] for m in picked] == ["EMSR1-1/s1rtc/EMSR1-1-1_s1rtc_pre_a.tif"]


def test_member_index_length_mismatch_is_an_error(get_data):
    index = _index("EMSR1-1/s1rtc/EMSR1-1-1_s1rtc_pre_a.tif", "EMSR1-1/s1rtc/EMSR1-1-2_s1rtc_pre_a.tif",
                   indexed=[[0, 1, "0" * 64]])
    with pytest.raises(SystemExit, match="1 byte ranges"):
        _select(get_data, index, "EMSR1-1-2")


def test_member_name_escaping_the_destination_is_rejected(get_data, tmp_path):
    member = {"shard": "s.tar", "tree": "geoid-flood", "rel": "../../escaped.tif",
              "offset": 0, "size": 1, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="escapes destination"):
        get_data.fetch_member("r", member, "rev", tmp_path, None)


def test_file_of_the_wrong_size_is_fetched_again(get_data, monkeypatch, tile_hub, tmp_path):
    payloads, ranges = tile_hub
    dest = tmp_path / "data"
    name = next(n for n in payloads if f"/{TILE_B}_" in n)
    target = dest / "geoid-flood" / name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"truncated")
    _run_tiles(get_data, monkeypatch, dest, ["--tile", TILE_B])
    assert target.read_bytes() == payloads[name]
    assert len(ranges) == 1


def test_list_fetches_nothing_and_creates_nothing(get_data, monkeypatch, tile_hub, tmp_path, capsys):
    _, ranges = tile_hub
    dest = tmp_path / "data"
    _run_tiles(get_data, monkeypatch, dest, ["--tile", TILE_A, "--list"])
    assert not ranges and not dest.exists()
    assert TILE_A in capsys.readouterr().out


def test_tile_mode_refuses_a_destination_from_another_revision(get_data, monkeypatch, tile_hub, tmp_path):
    dest = tmp_path / "data"
    _run_tiles(get_data, monkeypatch, dest, ["--tile", TILE_A, "--revision", "v1.0"])
    with pytest.raises(SystemExit, match="holds GEOID-Flood v1.0, not v1.1"):
        _run_tiles(get_data, monkeypatch, dest, ["--tile", TILE_A])


def test_transient_failures_are_retried_then_reported(get_data, monkeypatch, tile_hub, tmp_path):
    _, ranges = tile_hub
    real = get_data.fetch_range
    calls = {"n": 0}

    def flaky(*args):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("429")
        return real(*args)

    monkeypatch.setattr(get_data, "fetch_range", flaky)
    monkeypatch.setattr(get_data.time, "sleep", lambda s: None)
    _run_tiles(get_data, monkeypatch, tmp_path / "data", ["--tile", TILE_B])
    assert calls["n"] == 2

    def always_unavailable(*args):
        raise ValueError("503")

    monkeypatch.setattr(get_data, "fetch_range", always_unavailable)
    with pytest.raises(SystemExit):
        _run_tiles(get_data, monkeypatch, tmp_path / "other", ["--tile", TILE_B])
