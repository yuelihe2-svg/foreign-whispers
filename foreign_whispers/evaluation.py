"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""

import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def _safe_mean(values: list[float]) -> float:
    """
    Return the mean of a list, or 0.0 for an empty list.
    Keeps scorecard metrics stable when a clip has no valid segments.
    """
    return _stats.mean(values) if values else 0.0


def _score_from_error(mean_error_s: float) -> float:
    """
    Convert mean timing error into a 0-100 timing score.
    A 0s error scores 100; errors around 2s or larger approach 0.
    """
    return max(0.0, min(100.0, 100.0 * (1.0 - mean_error_s / 2.0)))


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """
    Return a multi-dimensional alignment quality report for one clip.

    The report combines timing accuracy, stretch/naturalness risk,
    translation retry pressure, and cumulative drift into one scorecard.。

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
        timing_accuracy_score: 0-100 timing score based on mean duration error.
        naturalness_score: 0-100 score penalizing severe stretching and failures.
        intelligibility_proxy_score: 0-100 proxy penalizing heavy stretch and failed segments.
        semantic_fidelity_proxy_score: 0-100 proxy penalizing REQUEST_SHORTER pressure.
        overall_quality_score: Weighted 0-100 aggregate score.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch": 0.0,
            "n_gap_shifts": 0,
            "n_translation_retries": 0,
            "total_cumulative_drift_s": 0.0,
            "timing_accuracy_score": 100.0,
            "naturalness_score": 100.0,
            "intelligibility_proxy_score": 100.0,
            "semantic_fidelity_proxy_score": 100.0,
            "overall_quality_score": 100.0,
        }

    n = max(len(metrics), 1)

    errors = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    mean_error = _safe_mean(errors)

    severe_stretch_count = sum(1 for a in aligned if a.stretch_factor > 1.4)
    failed_count = sum(1 for a in aligned if a.action == AlignAction.FAIL)
    gap_shift_count = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)

    retry_count = sum(
        1 for m in metrics
        if decide_action(m) == AlignAction.REQUEST_SHORTER
    )

    drift = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    pct_severe = 100.0 * severe_stretch_count / n
    pct_failed = 100.0 * failed_count / n
    pct_retry = 100.0 * retry_count / n

    # Timing accuracy primarily measures duration fit.
    timing_accuracy_score = _score_from_error(mean_error)

    # Naturalness is hurt by aggressive stretching and failed segments.
    naturalness_score = max(
        0.0,
        100.0 - (pct_severe * 1.5) - (pct_failed * 3.0),
    )

    # Without round-trip STT, use stretch/failure as an intelligibility proxy.
    intelligibility_proxy_score = max(
        0.0,
        100.0 - pct_severe - (pct_failed * 4.0),
    )

    # REQUEST_SHORTER pressure is a proxy for semantic fidelity risk.
    semantic_fidelity_proxy_score = max(
        0.0,
        100.0 - (pct_retry * 1.25),
    )

    # Weighted scorecard; timing matters most for dubbing.
    overall_quality_score = (
        timing_accuracy_score * 0.40
        + naturalness_score * 0.25
        + intelligibility_proxy_score * 0.20
        + semantic_fidelity_proxy_score * 0.15
    )

    return {
        "mean_abs_duration_error_s": round(mean_error, 3),
        "pct_severe_stretch": round(pct_severe, 1),
        "n_gap_shifts": gap_shift_count,
        "n_translation_retries": retry_count,
        "total_cumulative_drift_s": round(drift, 3),
        "timing_accuracy_score": round(timing_accuracy_score, 1),
        "naturalness_score": round(naturalness_score, 1),
        "intelligibility_proxy_score": round(intelligibility_proxy_score, 1),
        "semantic_fidelity_proxy_score": round(semantic_fidelity_proxy_score, 1),
        "overall_quality_score": round(overall_quality_score, 1),
    }