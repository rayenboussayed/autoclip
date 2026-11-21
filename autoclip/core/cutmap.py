"""Monotone mapping between the original timeline and the optimized (silence-removed) one."""

from __future__ import annotations

import hashlib
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field

CutMapVersion = int
_SCHEMA: CutMapVersion = 1


@dataclass(frozen=True, slots=True)
class CutMap:
    """Kept intervals, in original-source seconds, and both timeline projections."""

    duration: float
    intervals: tuple[tuple[float, float], ...]
    version: CutMapVersion = _SCHEMA
    _starts: tuple[float, ...] = field(init=False, repr=False, compare=False)
    _ends: tuple[float, ...] = field(init=False, repr=False, compare=False)
    _prefix: tuple[float, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("the duration must be positive")
        if self.intervals:
            previous = -1.0
            for start, end in self.intervals:
                if start < 0 or end <= start or end > self.duration:
                    raise ValueError(f"kept interval {start}-{end} is outside 0-{self.duration}")
                if start < previous:
                    raise ValueError("kept intervals must be sorted and disjoint")
                previous = end
        object.__setattr__(self, "_starts", tuple(start for start, _ in self.intervals))
        object.__setattr__(self, "_ends", tuple(end for _, end in self.intervals))
        total, prefix = 0.0, [0.0]
        for start, end in self.intervals:
            total += end - start
            prefix.append(total)
        object.__setattr__(self, "_prefix", tuple(prefix))

    @property
    def total_kept(self) -> float:
        """The length of the optimized timeline in seconds."""
        return sum(end - start for start, end in self.intervals)

    @classmethod
    def identity(cls, duration: float) -> CutMap:
        """The map for a source that was not altered."""
        return cls(duration, ((0.0, duration),))

    @classmethod
    def from_intervals(cls, duration: float, intervals: list[tuple[float, float]]) -> CutMap:
        """A validated map from the kept intervals of a silence pass."""
        return cls(duration, tuple((float(start), float(end)) for start, end in intervals))

    @property
    def fingerprint(self) -> str:
        """A stable identity for cache keys and incomplete-artifact manifests."""
        payload = repr((self.version, round(self.duration, 3), tuple((round(a, 3), round(b, 3)) for a, b in self.intervals)))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def chunked_pieces(self, seconds: float = 300.0) -> list[list[tuple[float, float, float]]]:
        """Whisper-sized chunks of ``(source_start, source_end, optimized_start)`` pieces, packed without exceeding ``seconds``."""
        chunks: list[list[tuple[float, float, float]]] = []
        current: list[tuple[float, float, float]] = []
        total = 0.0
        optimized = 0.0
        for source_start, source_end in self.intervals:
            remaining, start = source_end - source_start, source_start
            while remaining > 1e-9:
                if current and total >= seconds - 1e-9:
                    chunks.append(current)
                    current, total = [], 0.0
                capacity = seconds - total
                take = min(remaining, capacity)
                current.append((start, start + take, optimized))
                start += take
                remaining -= take
                optimized += take
                total += take
        if current:
            chunks.append(current)
        return chunks

    def source_to_optimized(self, time: float) -> float:
        """Map a source time; points inside removed spans map to the following kept edge."""
        clamped = min(max(time, 0.0), self.duration)
        index = bisect_right(self._starts, clamped) - 1
        if index >= 0 and clamped <= self._ends[index]:
            return self._prefix[index] + clamped - self._starts[index]
        pos = bisect_left(self._starts, clamped)
        return self._prefix[pos] if pos < len(self._prefix) else self._prefix[-1]

    def optimized_to_source(self, time: float) -> float:
        """Map an optimized time back to the source timeline."""
        clamped = min(max(time, 0.0), self.total_kept)
        if not self._starts:
            return 0.0
        index = bisect_right(self._prefix, clamped) - 1
        index = max(index, 0)
        return self._starts[index] + clamped - self._prefix[index]

    def source_span_to_optimized(self, start: float, end: float) -> tuple[float, float]:
        """Map a source span, keeping it continuous in optimized time."""
        mapped_start, mapped_end = self.source_to_optimized(start), self.source_to_optimized(end)
        return (mapped_start, max(mapped_start, mapped_end))


@dataclass(frozen=True, slots=True)
class MeasuredTimeline:
    """The real timeline of an assembled file, aligned piece by piece with the nominal cut map.

    Each kept interval is encoded as its own MP4; CFR frame rounding and AAC padding make the
    piece longer than its nominal source width, so the concatenated file is longer than
    :attr:`CutMap.total_kept`.  Transcripts and subtitles are cached on the nominal clock, while
    captions are burned into the real file - this monotone map is the single translation point.
    It carries no cache identity: the cached rows stay nominal and are remapped once per run.
    """

    intervals: tuple[tuple[float, float], ...]
    nominal: tuple[float, ...]  # cumulative nominal starts, one more entry than intervals
    real: tuple[float, ...]  # cumulative measured starts, one more entry than intervals

    @classmethod
    def from_durations(cls, intervals: Sequence[tuple[float, float]], durations: Sequence[float]) -> MeasuredTimeline:
        """Build from kept source spans and the measured duration of each encoded piece."""
        if not intervals or len(intervals) != len(durations):
            raise ValueError("one measured duration is required per kept interval")
        nominal, real = [0.0], [0.0]
        for (start, end), duration in zip(intervals, durations, strict=True):
            if end <= start or duration <= 0:
                raise ValueError("kept intervals and their measured durations must be positive")
            nominal.append(nominal[-1] + end - start)
            real.append(real[-1] + duration)
        return cls(tuple((float(a), float(b)) for a, b in intervals), tuple(nominal), tuple(real))

    @classmethod
    def from_cutmap(cls, cut: CutMap) -> MeasuredTimeline:
        """The nominal clock of a source that was not reassembled (measured width == nominal)."""
        if not cut.intervals:
            return cls.from_durations(((0.0, cut.duration),), (cut.duration,))
        return cls.from_durations(cut.intervals, [end - start for start, end in cut.intervals])

    @property
    def last(self) -> float:
        """The real duration of the assembled file."""
        return self.real[-1]

    def _piece(self, t: float, edges: tuple[float, ...]) -> int:
        """Index of the piece containing ``t`` on ``edges``, clamped to the available pieces."""
        return min(max(bisect_right(edges, t) - 1, 0), len(self.intervals) - 1)

    def to_real(self, t: float) -> float:
        """Nominal optimized time -> real time in the assembled file."""
        i = self._piece(t, self.nominal)
        start, end = self.intervals[i]
        local = min(max(t - self.nominal[i], 0.0), end - start)
        return self.real[i] + local * (self.real[i + 1] - self.real[i]) / (end - start)

    def to_nominal(self, t: float) -> float:
        """Real time in the assembled file -> nominal optimized time."""
        i = self._piece(t, self.real)
        start, end = self.intervals[i]
        duration = self.real[i + 1] - self.real[i]
        local = min(max(t - self.real[i], 0.0), duration)
        return self.nominal[i] + local * (end - start) / duration

    def to_source(self, t: float) -> float:
        """Real time in the assembled file -> original source time."""
        i = self._piece(t, self.real)
        start, end = self.intervals[i]
        duration = self.real[i + 1] - self.real[i]
        local = min(max(t - self.real[i], 0.0), duration)
        return start + local * (end - start) / duration

    def remap_window(self, start: float, end: float) -> tuple[float, float]:
        """A nominal window as a real window in the assembled file."""
        return self.to_real(start), self.to_real(end)
