import pytest

from audio_transcribator.benchmarks.transcription import (
    calculate_text_metrics,
    extract_reference_text,
    normalize_text_for_metric,
)


def test_normalize_text_for_metric_lowercases_and_removes_punctuation():
    assert normalize_text_for_metric("Ёжик, ПРИВЕТ!") == "ежик привет"


def test_text_metrics_perfect_match():
    metrics = calculate_text_metrics("привет мир", "привет мир")

    assert metrics["wer"] == pytest.approx(0.0)
    assert metrics["cer"] == pytest.approx(0.0)
    assert metrics["normalized_wer"] == pytest.approx(0.0)
    assert metrics["normalized_cer"] == pytest.approx(0.0)


def test_text_metrics_word_error():
    metrics = calculate_text_metrics("привет большой мир", "привет мир")

    assert metrics["wer"] == pytest.approx(1 / 3)
    assert metrics["normalized_wer"] == pytest.approx(1 / 3)


def test_text_metrics_raw_and_normalized_differ_on_punctuation():
    metrics = calculate_text_metrics("Привет, мир!", "привет мир")

    assert metrics["wer"] > 0
    assert metrics["normalized_wer"] == pytest.approx(0.0)


def test_extract_reference_text_from_speaker_segments():
    sample = {
        "speakers": [
            {"speaker_id": 1, "text": "первая реплика"},
            {"speaker_id": 2, "text": "вторая реплика"},
            {"speaker_id": 1, "text": ""},
        ]
    }

    assert extract_reference_text(sample) == "первая реплика вторая реплика"
