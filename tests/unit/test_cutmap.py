"""Timeline mapping for videos whose silent spans are removed."""

import pytest

from autoclip.core.cutmap import CutMap, MeasuredTimeline


def test_identity_map_round_trips_without_shift() -> None:
    cut = CutMap.identity(10.0)
    assert cut.total_kept == 10.0
    assert cut.source_to_optimized(3.5) == 3.5
    assert cut.optimized_to_source(7.0) == 7.0
    assert cut.source_span_to_optimized(1.0, 9.0) == (1.0, 9.0)


def test_map_skips_removed_spans() -> None:
    cut = CutMap.from_intervals(20.0, [(0.0, 4.0), (7.0, 15.0)])
    assert cut.total_kept == 12.0
    assert cut.source_to_optimized(3.0) == 3.0
    assert cut.source_to_optimized(5.0) == 4.0  # removed middle maps to the next kept edge
    assert cut.source_to_optimized(10.0) == 7.0
    assert cut.optimized_to_source(6.5) == 9.5
    assert cut.source_span_to_optimized(3.0, 10.0) == (3.0, 7.0)


def test_fingerprint_changes_with_the_kept_set_and_version() -> None:
    left = CutMap.from_intervals(10.0, [(0.0, 4.0), (6.0, 10.0)])
    right = CutMap.from_intervals(10.0, [(0.0, 4.0), (7.0, 10.0)])
    assert left.fingerprint == CutMap.from_intervals(10.0, [(0.0, 4.0), (6.0, 10.0)]).fingerprint
    assert left.fingerprint != right.fingerprint
    assert CutMap(10.0, left.intervals, version=2).fingerprint != left.fingerprint


def test_chunked_pieces_bound_the_whisper_input() -> None:
    long = CutMap.identity(700.0).chunked_pieces(300.0)
    assert [sum(end - start for start, end, _ in chunk) for chunk in long] == [300.0, 300.0, 100.0]
    cut = CutMap.from_intervals(20.0, [(0.0, 4.0), (10.0, 15.0)])
    assert cut.chunked_pieces(3.0) == [
        [(0.0, 3.0, 0.0)],
        [(3.0, 4.0, 3.0), (10.0, 12.0, 4.0)],
        [(12.0, 15.0, 6.0)],
    ]


def test_map_rejects_malformed_intervals() -> None:
    with pytest.raises(ValueError):
        CutMap(10.0, ((3.0, 2.0),))
    with pytest.raises(ValueError):
        CutMap(10.0, ((0.0, 11.0),))
    with pytest.raises(ValueError):
        CutMap(10.0, ((2.0, 5.0), (4.0, 6.0)))


def test_cutmap_rejects_bad_ranges_and_empty_maps_behave_sensibly() -> None:
    with pytest.raises(ValueError):
        CutMap(0.0, ())
    assert CutMap(10.0, ()).total_kept == 0.0
    assert CutMap.identity(10.0)
    with pytest.raises(ValueError):
        CutMap(10.0, ((-1.0, 1.0),))
    with pytest.raises(ValueError):
        CutMap(10.0, ((1.0, 1.0),))
    with pytest.raises(ValueError):
        CutMap(10.0, ((0.0, 11.0),))
    assert CutMap(10.0, ((0.0, 2.0), (5.0, 8.0))).total_kept == 5.0


def test_cutmap_edge_time_mappings() -> None:
    cut = CutMap.from_intervals(20.0, [(5.0, 8.0), (12.0, 14.0)])
    assert cut.source_to_optimized(0.0) == 0.0
    assert cut.source_to_optimized(9.0) == 3.0
    assert cut.source_to_optimized(30.0) == 5.0
    assert cut.optimized_to_source(0.0) == 5.0
    assert cut.optimized_to_source(4.0) == 13.0
    assert CutMap(10.0, ()).optimized_to_source(5.0) == 0.0


def test_chunked_pieces_returns_no_chunks_for_an_empty_kept_set() -> None:
    assert CutMap(10.0, ()).chunked_pieces(30.0) == []


def test_measured_timeline_maps_nominal_real_and_source_times() -> None:
    """A piece that encodes longer than its nominal width stretches only its own span."""
    timeline = MeasuredTimeline.from_durations(((0.0, 4.0), (10.0, 15.0)), (4.4, 5.5))
    assert timeline.nominal == (0.0, 4.0, 9.0)
    assert timeline.real == (0.0, 4.4, 9.9)
    assert timeline.last == 9.9
    assert timeline.to_real(2.0) == pytest.approx(2.2)
    assert timeline.to_real(6.5) == pytest.approx(4.4 + 2.5 * 1.1)
    assert timeline.to_nominal(timeline.to_real(6.5)) == pytest.approx(6.5)
    # a real time just inside the second piece is near source 11 s
    assert timeline.to_source(4.4 + 1.1) == pytest.approx(11.0)
    remapped = timeline.remap_window(0.0, 9.0)
    assert remapped[0] == 0.0 and remapped[1] == pytest.approx(9.9)


def test_measured_timeline_identity_and_validation() -> None:
    """An unassembled source uses a nominal clock; malformed measurements are rejected."""
    assert MeasuredTimeline.from_cutmap(CutMap(10.0, ())).last == 10.0
    assert MeasuredTimeline.from_cutmap(CutMap.identity(10.0)).to_real(4.0) == 4.0
    with pytest.raises(ValueError):
        MeasuredTimeline.from_durations((), ())
    with pytest.raises(ValueError):
        MeasuredTimeline.from_durations(((0.0, 4.0),), (4.0, 5.0))
    with pytest.raises(ValueError):
        MeasuredTimeline.from_durations(((0.0, 4.0),), (0.0,))
    with pytest.raises(ValueError):
        MeasuredTimeline.from_durations(((2.0, 1.0),), (1.0,))
    clamped = MeasuredTimeline.from_durations(((0.0, 4.0),), (5.0,))
    assert clamped.to_real(-1.0) == 0.0 and clamped.to_real(99.0) == 5.0
    assert clamped.to_nominal(-1.0) == 0.0 and clamped.to_nominal(99.0) == 4.0
    assert clamped.to_source(-1.0) == 0.0 and clamped.to_source(99.0) == 4.0
