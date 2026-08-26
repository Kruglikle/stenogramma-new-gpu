import pytest

from audio_transcribator.benchmarks.diarization import Turn, calculate_diarization_metrics, turns_to_annotation


def score(reference_turns, hypothesis_turns):
    reference = turns_to_annotation(reference_turns, uri="test")
    hypothesis = turns_to_annotation(hypothesis_turns, uri="test")
    return calculate_diarization_metrics(reference, hypothesis)


def test_perfect_match_has_zero_der_and_jer():
    metrics = score(
        [Turn("A", 0.0, 1.0), Turn("B", 1.0, 2.0)],
        [Turn("A", 0.0, 1.0), Turn("B", 1.0, 2.0)],
    )

    assert metrics["der"] == pytest.approx(0.0)
    assert metrics["jer"] == pytest.approx(0.0)


def test_different_names_for_same_speakers_are_mapped():
    metrics = score(
        [Turn("alice", 0.0, 1.0), Turn("bob", 1.0, 2.0)],
        [Turn("speaker_0", 0.0, 1.0), Turn("speaker_1", 1.0, 2.0)],
    )

    assert metrics["der"] == pytest.approx(0.0)


def test_missed_speech_is_reported():
    metrics = score(
        [Turn("A", 0.0, 2.0)],
        [Turn("A", 0.0, 1.0)],
    )

    assert metrics["der"] == pytest.approx(0.5)
    assert metrics["miss"] == pytest.approx(0.5)
    assert metrics["false_alarm"] == pytest.approx(0.0)


def test_false_alarm_speech_is_reported():
    metrics = score(
        [Turn("A", 0.0, 1.0)],
        [Turn("A", 0.0, 1.0), Turn("A", 1.0, 2.0)],
    )

    assert metrics["der"] == pytest.approx(1.0)
    assert metrics["false_alarm"] == pytest.approx(1.0)
    assert metrics["miss"] == pytest.approx(0.0)


def test_speaker_confusion_is_reported():
    metrics = score(
        [Turn("A", 0.0, 1.0), Turn("B", 1.0, 2.0), Turn("A", 2.0, 3.0)],
        [Turn("A", 0.0, 1.0), Turn("A", 1.0, 2.0), Turn("A", 2.0, 3.0)],
    )

    assert metrics["der"] == pytest.approx(1.0 / 3.0)
    assert metrics["speaker_confusion"] == pytest.approx(1.0 / 3.0)


def test_overlapping_speech_is_scored_without_skipping_overlap():
    metrics = score(
        [Turn("A", 0.0, 2.0), Turn("B", 1.0, 3.0)],
        [Turn("A", 0.0, 3.0)],
    )

    assert metrics["der"] > 0.0
    assert metrics["reference_speech_seconds"] == pytest.approx(4.0)
