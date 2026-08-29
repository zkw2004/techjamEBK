"""D6 acceptance tests for the proposing agent's static recsys priors."""

from pathlib import Path

KNOWLEDGE = Path(__file__).parents[1] / "agent" / "knowledge.md"


def test_knowledge_covers_every_required_prior_with_a_rationale():
    text = KNOWLEDGE.read_text(encoding="utf-8")
    required = (
        "FM",
        "DeepFM",
        "LightGBM lambdarank",
        "Multi-task heads",
        "BPR / pairwise",
        "Negative sampling",
        "Time decay",
        "Empirical-Bayes smoothing",
        "Exposure debiasing (randomised slice)",
        "Blending",
    )

    for prior in required:
        marker = f"- **{prior}** — "
        assert marker in text
        rationale = text.split(marker, maxsplit=1)[1].split("\n- **", maxsplit=1)[0]
        assert len(rationale.split()) >= 8

    assert "TODO" not in text


def test_leakage_rules_and_feature_policy_are_emphasised():
    text = KNOWLEDGE.read_text(encoding="utf-8")

    assert "## Leakage rules — non-negotiable" in text
    assert "**Never shuffle or k-fold.**" in text
    assert "**Never use a same-row post-exposure signal as an input feature.**" in text
    assert "**Never fit on the official validation window**" in text
    assert "| Column | As input feature (same row) |" in text
