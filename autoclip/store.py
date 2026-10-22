"""Global storage: a SQLite database (WAL) plus a two-tier cache (bounded RAM LRU in front of SQLite).

Each process opens its own connection to the same file. Large data (transcript words, candidate windows,
cached arrays) lives on disk; callers load only the rows they need so RAM stays free.
"""

from __future__ import annotations

import io
import sqlite3
import time
import zlib
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .models import Moment, Segment, Word
from .resources import MB, ResourceMonitor

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments(id INTEGER PRIMARY KEY, start REAL, end REAL, text TEXT,
                                    boundary INTEGER);
CREATE TABLE IF NOT EXISTS words(start REAL, end REAL, text TEXT, pseudo INTEGER);
CREATE INDEX IF NOT EXISTS ix_words_start ON words(start);
CREATE TABLE IF NOT EXISTS clips(id INTEGER PRIMARY KEY, kind TEXT, scene TEXT, n INTEGER, start REAL, end REAL,
                                 score REAL, llm_score REAL, title TEXT, text TEXT, status TEXT, file TEXT,
                                 width INTEGER, height INTEGER, trimmed REAL, wall REAL, error TEXT);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v BLOB NOT NULL, size INTEGER, atime REAL);
CREATE INDEX IF NOT EXISTS ix_kv_atime ON kv(atime);
"""
#: Words and segments are shorter than this, so a window query only needs to look this far back.
MAX_SPAN = 120.0


class Store:
    """The SQLite database: transcript, clip results and cached blobs."""

    def __init__(self, path: Path, fresh: bool = False, cache_kb: int = 16384) -> None:
        """Open the database, optionally deleting any previous one (and its WAL/SHM files) first."""
        self.path = path
        if fresh:
            for suffix in ("", "-wal", "-shm", "-journal"):
                Path(f"{path}{suffix}").unlink(missing_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=60, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        for pragma in (
            "journal_mode=WAL",
            "synchronous=NORMAL",
            "temp_store=FILE",  # sorts / temp tables spill to disk, not RAM
            f"cache_size=-{cache_kb}",
            "busy_timeout=60000",
        ):
            self.conn.execute(f"PRAGMA {pragma}")
        self.conn.executescript(SCHEMA)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """``BEGIN IMMEDIATE`` ... ``COMMIT`` around a block (rolled back on error)."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    def close(self) -> None:
        """Checkpoint the WAL (so the file shrinks) and close."""
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.conn.close()

    # -- transcript -----------------------------------------------------------------------------
    def put_segments(self, segments: list[Segment]) -> None:
        """Replace the stored transcript, precomputing each segment's sentence-boundary flag."""
        rows, words = [], []
        for idx, seg in enumerate(segments):
            gap = segments[idx + 1].start - seg.end if idx + 1 < len(segments) else None
            rows.append((idx, seg.start, seg.end, seg.text, int(is_boundary(seg.text, gap))))
            words += [(w.start, w.end, w.text, int(w.pseudo)) for w in seg.words]
        with self.transaction():
            self.conn.execute("DELETE FROM segments")
            self.conn.execute("DELETE FROM words")
            self.conn.executemany("INSERT INTO segments VALUES (?,?,?,?,?)", rows)
            self.conn.executemany("INSERT INTO words VALUES (?,?,?,?)", words)

    def light_segments(self) -> list[tuple[float, float, str, bool]]:
        """``(start, end, text, ends_sentence)`` of every segment - without the (large) word table."""
        rows = self.conn.execute("SELECT start, end, text, boundary FROM segments ORDER BY id")
        return [(r["start"], r["end"], r["text"], bool(r["boundary"])) for r in rows]

    def segment_text(self, i: int, j: int) -> str:
        """Joined text of segments ``i..j`` (inclusive)."""
        rows = self.conn.execute("SELECT text FROM segments WHERE id BETWEEN ? AND ? ORDER BY id", (i, j))
        return " ".join(r["text"] for r in rows)

    def window_text(self, a: float, b: float) -> str:
        """Joined text of every segment overlapping ``[a, b]``."""
        rows = self.conn.execute("SELECT text FROM segments WHERE start >= ? AND start < ? AND end > ? ORDER BY id", (a - MAX_SPAN, b, a))
        return " ".join(r["text"] for r in rows)

    def window_words(self, a: float, b: float) -> list[Word]:
        """Words overlapping ``[a, b]`` with times clamped to the window (absolute seconds)."""
        rows = self.conn.execute(
            "SELECT start, end, text, pseudo FROM words WHERE start >= ? AND start < ? AND end > ? ORDER BY start, rowid",
            (a - MAX_SPAN, b, a),
        )
        return [Word(max(r["start"], a), min(r["end"], b), r["text"], bool(r["pseudo"])) for r in rows]

    # -- clips ----------------------------------------------------------------------------------
    def save_moment(self, kind: str, scene: str, n: int, m: Moment) -> int:
        """Register a selected clip as ``pending`` and return its id."""
        cur = self.conn.execute(
            "INSERT INTO clips(kind, scene, n, start, end, score, llm_score, title, text, status) VALUES (?,?,?,?,?,?,?,?,?, 'pending')",
            (kind, scene, n, m.start, m.end, m.score, m.llm_score, m.title, m.text),
        )
        assert cur.lastrowid is not None
        return cur.lastrowid

    def finish_clip(
        self,
        clip_id: int,
        *,
        ok: bool,
        file: str = "",
        size: tuple[int, int] = (0, 0),
        trimmed: float = 0.0,
        wall: float = 0.0,
        error: str = "",
    ) -> None:
        """Record the render outcome of a clip."""
        self.conn.execute(
            "UPDATE clips SET status=?, file=?, width=?, height=?, trimmed=?, wall=?, error=? WHERE id=?",
            ("done" if ok else "failed", file, size[0], size[1], trimmed, wall, error, clip_id),
        )

    def clips(self, kind: str | None = None) -> list[dict[str, Any]]:
        """Clip rows (optionally of one kind) in creation order, as dicts."""
        where, args = ("WHERE kind=?", (kind,)) if kind else ("", ())
        return [dict(r) for r in self.conn.execute(f"SELECT * FROM clips {where} ORDER BY id", args)]  # noqa: S608

    # -- cache blobs ----------------------------------------------------------------------------
    def kv_get(self, key: str) -> bytes | None:
        """A cached blob (its access time is refreshed) or ``None``."""
        row = self.conn.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        if row is None:
            return None
        self.conn.execute("UPDATE kv SET atime=? WHERE k=?", (time.time(), key))
        return bytes(row["v"])

    def kv_put(self, key: str, value: bytes) -> None:
        """Store or replace a blob."""
        self.conn.execute("INSERT OR REPLACE INTO kv(k, v, size, atime) VALUES (?,?,?,?)", (key, value, len(value), time.time()))

    def kv_evict(self, max_bytes: int) -> None:
        """Delete least-recently-used blobs until the total size is at most ``max_bytes``."""
        excess = self.conn.execute("SELECT COALESCE(SUM(size), 0) - ? FROM kv", (max_bytes,)).fetchone()[0]
        for row in self.conn.execute("SELECT k, size FROM kv ORDER BY atime").fetchall():
            if excess <= 0:
                break
            self.conn.execute("DELETE FROM kv WHERE k=?", (row["k"],))
            excess -= row["size"]


def is_boundary(text: str, gap_to_next: float | None) -> bool:
    """Does a segment end a sentence? (terminal punctuation, a pause >= 0.6 s, or the last segment)."""
    ender = (
        text.strip()
        .rstrip(" \"'\u201d\u2019)\u00bb\u300d\u300f")
        .endswith((".", "?", "!", "\u2026", "\u3002", "\uff01", "\uff1f", "\u061f", "\u0964", "\u0965"))
    )
    return ender or gap_to_next is None or gap_to_next >= 0.6


class MemoryLRU:
    """A dictionary of bytes limited by total size; least recently used entries are evicted first."""

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.size = 0
        self._items: OrderedDict[str, bytes] = OrderedDict()

    def get(self, key: str) -> bytes | None:
        """The value, marked most recently used."""
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key: str, value: bytes) -> None:
        """Insert ``value``; one larger than the whole budget is not kept."""
        old = self._items.pop(key, None)
        self.size -= len(old) if old is not None else 0
        if len(value) <= self.budget:
            self._items[key] = value
            self.size += len(value)
        self.resize(self.budget)

    def resize(self, budget: int) -> None:
        """Set a new budget, evicting immediately if it shrank."""
        self.budget = budget
        while self.size > budget:
            self.size -= len(self._items.popitem(last=False)[1])


class Cache:
    """Bytes / NumPy-array cache: RAM LRU in front of the SQLite ``kv`` table.

    The RAM budget follows :meth:`ResourceMonitor.cache_budget`, re-evaluated by :meth:`trim` after every
    traced operation.
    """

    def __init__(self, store: Store, monitor: ResourceMonitor) -> None:
        self.store, self.monitor = store, monitor
        self.mem = MemoryLRU(monitor.cache_budget())
        monitor.register_trim_hook(self.trim)

    def close(self) -> None:
        """Detach from the monitor (call before closing the :class:`Store`)."""
        self.monitor.unregister_trim_hook(self.trim)

    def trim(self) -> None:
        """Re-balance: resize the RAM tier to the current budget and cap the blob space in SQLite."""
        self.mem.resize(self.monitor.cache_budget())
        self.store.kv_evict(self.monitor.storage.cache_disk_mb * MB)

    def memoize_array(self, key: str, producer: Callable[[], NDArray[Any]]) -> NDArray[Any]:
        """The cached array for ``key``, computing (and caching, zlib-compressed) it with ``producer`` on a miss."""
        blob = self.mem.get(key)
        if blob is None:
            blob = self.store.kv_get(key)
        if blob is None:
            buf = io.BytesIO()
            np.save(buf, producer(), allow_pickle=False)
            blob = zlib.compress(buf.getvalue(), 1)
            self.store.kv_put(key, blob)
        self.mem.put(key, blob)
        arr: NDArray[Any] = np.load(io.BytesIO(zlib.decompress(blob)), allow_pickle=False)
        return arr
