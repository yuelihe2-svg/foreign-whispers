"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.
    Implementation: Rule-based approach
    Uses multi-stage text reduction to fit within the target duration budget.
    """
    logger.info(
        "get_shorter_translations called for %.1fs budget (%d chars baseline)",
        target_duration_s,
        len(baseline_es),
    )

    candidates = []

    # Heuristic: ~15 chars per second for Spanish
    target_char_count = int(target_duration_s * 15.0)

    # Stage 1: Remove common fillers and hesitations
    fillers = ["bueno, ", "sabes, ", "pues, ", "eh, ", "um, ", "quiero decir, "]
    text_no_fillers = baseline_es
    for filler in fillers:
        # Case-insensitive removal
        text_no_fillers = text_no_fillers.replace(filler, "").replace(filler.capitalize(), "")

    if text_no_fillers != baseline_es:
        candidates.append(TranslationCandidate(
            text=text_no_fillers,
            char_count=len(text_no_fillers),
            brevity_rationale="Removed filler words"
        ))

    # Stage 2: Replace long phrases with shorter synonyms
    replacements = {
        "en este momento": "ahora",         # at this moment -> now
        "en este punto": "ahora",           # at this point -> now
        "por supuesto": "claro",            # of course -> sure
        "sin embargo": "pero",              # however -> but
        "es necesario que": "hay que",      # it is necessary to -> must
        "con el fin de": "para",            # in order to -> for
        "de acuerdo con": "según",          # according to -> per
        "a pesar de que": "aunque",         # despite the fact that -> although
        "por lo tanto": "así",              # therefore -> so
        "en la mayoría de los casos": "generalmente", # in most cases -> generally
        "yo creo que": "creo que",          # I believe that -> believe that (drop pronoun)
        "yo pienso que": "pienso que"       # I think that -> think that
    }

    text_replaced = text_no_fillers
    for old, new in replacements.items():
        text_replaced = text_replaced.replace(old, new).replace(old.capitalize(), new.capitalize())

    if text_replaced != text_no_fillers:
        candidates.append(TranslationCandidate(
            text=text_replaced,
            char_count=len(text_replaced),
            brevity_rationale="Replaced long phrases with shorter synonyms"
        ))

    # Stage 3: Aggressive modifier removal (Only if still too long)
    if len(text_replaced) > target_char_count + 5: # 5 char buffer
        text_aggressive = text_replaced
        adverbs_to_remove = ["muy ", "realmente ", "simplemente ", "absolutamente "]
        for adv in adverbs_to_remove:
            text_aggressive = text_aggressive.replace(adv, "")

        if text_aggressive != text_replaced and len(text_aggressive) > 0:
            candidates.append(TranslationCandidate(
                text=text_aggressive,
                char_count=len(text_aggressive),
                brevity_rationale="Aggressively removed non-essential adverbs"
            ))

    # Stage 4: Budget-aware word-boundary trimming as a final fallback.
    has_fitting_candidate = any(c.char_count <= target_char_count for c in candidates)

    if len(text_replaced) > target_char_count and not has_fitting_candidate:
        max_chars = max(12, target_char_count)
        trimmed_words = []
        current = ""

        for word in text_replaced.split():
            candidate_text = f"{current} {word}".strip() if current else word

            if len(candidate_text) > max_chars:
                break

            current = candidate_text
            trimmed_words.append(word)

        text_trimmed = " ".join(trimmed_words).strip()
        text_trimmed = text_trimmed.rstrip(",;:")

        # Only add a fallback if it is non-empty and meaningfully shorter.
        if text_trimmed and len(text_trimmed) < len(baseline_es):
            candidates.append(TranslationCandidate(
                text=text_trimmed,
                char_count=len(text_trimmed),
                brevity_rationale="Trimmed to duration budget on word boundary"
            ))



    # Sort candidates by length (shortest first)
    candidates.sort(key=lambda c: c.char_count)

    return candidates
