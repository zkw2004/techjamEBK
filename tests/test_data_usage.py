"""B13 acceptance: a shipped file neither joined nor excluded is caught by
name; the exclusion map with reasons is rendered into judge-visible output;
the current real manifest is fully accounted for."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools.data_usage import (
    EXCLUDED,
    SHIPPED_FILES,
    check_data_usage,
    main,
    render_report,
)

# --- The failure path (B13's actual acceptance criterion) -------------------


def test_fabricated_unjoined_unexcluded_file_is_caught():
    """A made-up 'shipped' file that appears in neither the source scan nor
    EXCLUDED must be named in the result. This is the deterministic failure
    path B13 requires — proving the checker actually catches a real gap,
    not just that the current all-green state stays green."""
    fabricated = "totally_fake_shipped_file_not_real.csv"
    result = check_data_usage(
        shipped_files=(fabricated,),
        excluded={},
        source_text="some source code that never mentions the fabricated file",
    )
    assert result == [fabricated]


def test_fabricated_file_among_real_ones_is_named_specifically():
    """Mix one fabricated gap in among otherwise-accounted-for files; only
    the gap should be reported, by name."""
    fabricated = "another_fake_shipped_file.csv"
    shipped = (*SHIPPED_FILES, fabricated)
    source_text = "\n".join(SHIPPED_FILES)  # every real file "joined", fabricated one is not
    result = check_data_usage(shipped_files=shipped, excluded=EXCLUDED, source_text=source_text)
    assert result == [fabricated]


def test_check_data_usage_returns_list_type():
    result = check_data_usage(
        shipped_files=("x.csv",), excluded={"x.csv": "reason"}, source_text=""
    )
    assert isinstance(result, list)
    assert result == []


# --- Exclusion logic ---------------------------------------------------------


def test_excluded_file_not_flagged_even_if_unjoined():
    result = check_data_usage(
        shipped_files=("only_excluded.csv",),
        excluded={"only_excluded.csv": "some documented reason"},
        source_text="",
    )
    assert result == []


def test_joined_file_not_flagged_even_without_exclusion_entry():
    result = check_data_usage(
        shipped_files=("joined_file.csv",),
        excluded={},
        source_text="path = DATA_DIR / 'joined_file.csv'",
    )
    assert result == []


def test_file_both_joined_and_excluded_is_fine():
    """Overlap is not an error — a file can be excluded historically and
    later wired in without anyone remembering to delete the entry; that's
    a cleanup nit, not a checker failure."""
    result = check_data_usage(
        shipped_files=("both.csv",),
        excluded={"both.csv": "stale reason"},
        source_text="reads both.csv somewhere",
    )
    assert result == []


# --- The real, current manifest ---------------------------------------------


def test_real_manifest_is_fully_accounted_for():
    """The actual shipped-file list must currently be all-green: every file
    is either found in the real pipeline source or has an EXCLUDED entry."""
    result = check_data_usage()
    assert result == [], f"unaccounted-for shipped files: {result}"


def test_manifest_has_exactly_six_files():
    assert len(SHIPPED_FILES) == 6
    assert len(set(SHIPPED_FILES)) == 6, "manifest must not contain duplicates"


def test_the_four_log_and_video_basic_files_are_joined_not_excluded():
    """These four are read directly by pipeline.data; they must NOT appear
    in EXCLUDED — an exclusion entry for a genuinely-joined file would be
    misleading in the judge-visible report."""
    joined_files = [
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
        "log_random_4_22_to_5_08_pure.csv",
        "video_features_basic_pure.csv",
    ]
    for filename in joined_files:
        assert filename in SHIPPED_FILES
        assert filename not in EXCLUDED, f"{filename} is joined; should not be in EXCLUDED"


def test_the_two_unwired_files_are_excluded_with_reasons():
    for filename in ("user_features_pure.csv", "video_features_statistic_pure.csv"):
        assert filename in EXCLUDED
        reason = EXCLUDED[filename]
        assert isinstance(reason, str)
        assert len(reason.strip()) > 20, f"{filename}'s EXCLUDED reason looks too thin"


def test_statistic_file_reason_cites_trap_3_and_naming_mismatch():
    """Ground-truth requirement: the video_features_statistic_pure.csv
    exclusion reason must cite Section 11 trap 3 and flag the naming
    mismatch with pipeline.features.EXCLUDED_SOURCES."""
    reason = EXCLUDED["video_features_statistic_pure.csv"]
    assert "trap 3" in reason.lower() or "Section 11" in reason
    assert "item_statistics_monthly" in reason
    assert "EXCLUDED_SOURCES" in reason


# --- Judge-visible output (printed report) -----------------------------------


def test_render_report_includes_every_shipped_file():
    report = render_report()
    for filename in SHIPPED_FILES:
        assert filename in report


def test_render_report_includes_exclusion_reasons():
    report = render_report()
    for filename, reason in EXCLUDED.items():
        assert filename in report
        # A representative fragment of the reason should show up verbatim.
        assert reason.split(".")[0][:30] in report


def test_render_report_names_unaccounted_file_in_fail_result():
    fabricated = "unaccounted_example.csv"
    report = render_report(
        shipped_files=(fabricated,), excluded={}, unaccounted=[fabricated]
    )
    assert "FAIL" in report
    assert fabricated in report


def test_render_report_says_ok_when_nothing_unaccounted():
    report = render_report(shipped_files=(), excluded={}, unaccounted=[])
    assert "OK" in report
    assert "FAIL" not in report


# --- Exit-code behaviour, unit-tested without shelling out -------------------


def test_main_returns_zero_on_the_real_clean_manifest():
    assert main([]) == 0


def test_main_returns_nonzero_when_a_gap_exists(monkeypatch):
    """Monkeypatch check_data_usage so main() sees a fabricated gap, and
    confirm it returns non-zero without needing a real unjoined file."""
    import tools.data_usage as data_usage_module

    def fake_check(*args, **kwargs):
        return ["fabricated_gap_file.csv"]

    monkeypatch.setattr(data_usage_module, "check_data_usage", fake_check)
    assert main([]) == 1


# --- One subprocess-level integration test for extra confidence -------------


def test_subprocess_invocation_exits_zero_and_prints_exclusion_map():
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "tools.data_usage"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "user_features_pure.csv" in result.stdout
    assert "video_features_statistic_pure.csv" in result.stdout
    assert "RESULT: OK" in result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
