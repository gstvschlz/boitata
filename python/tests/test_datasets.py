import hashlib
import io

import ceres as cs
import pytest

CSV = b"X,Y,V\n1,2,3\n"


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("CERES_DATA", str(tmp_path))
    monkeypatch.setitem(cs.datasets.FILES, "walker-lake/sample.csv", hashlib.sha256(CSV).hexdigest())
    return tmp_path


def test_fetch_checks_the_hash_and_caches(offline, monkeypatch):
    monkeypatch.setattr(cs.datasets.urllib.request, "urlopen", lambda url: io.BytesIO(CSV))
    path = cs.datasets.fetch("walker-lake/sample.csv")
    assert path.read_bytes() == CSV and path.is_relative_to(offline)

    def no_network(url):
        raise AssertionError("downloaded twice")

    monkeypatch.setattr(cs.datasets.urllib.request, "urlopen", no_network)
    assert len(cs.datasets.walker_lake()) == 1


def test_corrupt_download_is_an_error_and_leaves_nothing(offline, monkeypatch):
    monkeypatch.setattr(cs.datasets.urllib.request, "urlopen", lambda url: io.BytesIO(b"tampered"))
    with pytest.raises(cs.FileError, match="SHA-256"):
        cs.datasets.fetch("walker-lake/sample.csv")
    assert not any(p.is_file() for p in offline.rglob("*"))


def test_unknown_file():
    with pytest.raises(cs.InvalidInput):
        cs.datasets.fetch("nope.csv")


@pytest.mark.slow
def test_download_walker_lake():
    assert len(cs.datasets.walker_lake()) == 470
    assert cs.datasets.walker_lake_exhaustive().num_rows == 78_000
