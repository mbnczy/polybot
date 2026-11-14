"""The recording reader survives a recorder stopped mid-block."""

from __future__ import annotations

import gzip
import zlib

from scripts.analyze_ws_recording import _lines


def _member(text: str) -> bytes:
    return gzip.compress(text.encode("ascii"))


def test_every_member_is_read(tmp_path):
    p = tmp_path / "h.csv.gz"
    p.write_bytes(_member("a\nb\n") + _member("c\n"))
    assert list(_lines(p)) == ["a\n", "b\n", "c\n"]


def test_a_truncated_member_does_not_lose_the_ones_after_it(tmp_path):
    body = "".join(f"{i},line\n" for i in range(5000))
    cut = _member(body)[:-4000]                       # stopped mid-block
    p = tmp_path / "h.csv.gz"
    p.write_bytes(_member("first\n") + cut + _member("after\n"))
    lines = list(_lines(p))
    assert lines[0] == "first\n" and lines[-1] == "after\n"
    assert all(l.endswith("\n") and l.count(",") == 1 for l in lines[1:-1])


def test_the_hour_still_being_written_ends_early_without_error(tmp_path):
    body = "".join(f"{i},line\n" for i in range(5000))
    c = zlib.compressobj(9, zlib.DEFLATED, 31)
    open_member = c.compress(body.encode()) + c.flush(zlib.Z_SYNC_FLUSH)   # no trailer yet
    p = tmp_path / "h.csv.gz"
    p.write_bytes(open_member)
    assert list(_lines(p))[-1] == "4999,line\n"
