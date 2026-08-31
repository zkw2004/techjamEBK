"""Data-usage checker. Task B13.

Pre-flight check: every data file shipped in the starter kit / KuaiRand
download must be accounted for — either joined somewhere into the pipeline
source, or explicitly listed in `EXCLUDED` with a reason. Running with a
shipped file that is neither is a bug (an unused file nobody decided to
exclude) and must fail loudly, by name, before any experiment runs.

Design (see AGENT_PLAN.md Section 9.2 row B13, Section 11 trap 3):

- The manifest of shipped files (`SHIPPED_FILES`) is a **static constant
  list**, not a directory listing. This lets the checker run as a fast
  pre-flight without requiring the git-ignored dataset archive to be on
  disk at all — it inspects source code, not `pipeline.data.DATA_DIR`.
- "Joined into the pipeline" is a **static source-scan**: for each shipped
  filename, check whether that literal string appears anywhere in the
  source of `pipeline/data.py`, `pipeline/features.py`, or
  `pipeline/train.py`. This mirrors the static-source-scan technique
  `pipeline.features.leakage_check` already uses elsewhere in this
  codebase, rather than inventing a new one. It deliberately does not try
  to actually load and run the pipeline — that needs the real dataset
  archive and would make a pre-flight check slow and archive-dependent.
- `EXCLUDED` is a module-level `{filename: reason}` dict, extended by hand
  whenever a future task wires one of these files up (at which point its
  entry should be deleted — the checker will then correctly detect the
  file as "joined" via the source scan instead).

Invocation: ``python -m tools.data_usage`` (see `if __name__ == "__main__"`
below). Exits 0 when every shipped file is accounted for; exits 1 and names
the offending file(s) otherwise. The exclusion map, with reasons, is always
printed to stdout — this is the "run log" for a standalone pre-flight
script per B13's acceptance criteria (D7/the agent loop can capture or
redirect this output later; this module does not write to logs/run.jsonl
itself, that is Workstream A/D territory).
"""

from __future__ import annotations

import argparse
import inspect
import sys

from pipeline import data as pipeline_data

# --- The shipped-file manifest (static, Section 9.2 B13) --------------------
#
# Six files ship in the KuaiRand-Pure starter kit / download, matching the
# layout `pipeline.data.DATA_DIR` expects. Four are already named by
# `pipeline.data` constants — imported here rather than re-hardcoded, so this
# manifest cannot silently drift from the loader's actual source of truth.
# The remaining two (`user_features_pure.csv`,
# `video_features_statistic_pure.csv`) are not exported by any module yet,
# so they are hardcoded literals — the only two in this file.
SHIPPED_FILES: tuple[str, ...] = (
    *pipeline_data.LOG_FILES,  # log_standard_4_08_to_4_21_pure.csv, ..._4_22_to_5_08_pure.csv
    pipeline_data.RANDOM_LOG_FILE,  # log_random_4_22_to_5_08_pure.csv
    "video_features_basic_pure.csv",  # read directly in pipeline.data.load() for author_id/tag
    "user_features_pure.csv",
    "video_features_statistic_pure.csv",
)

# --- Modules scanned for "is this file joined anywhere" ---------------------
#
# pipeline.data.py is the obvious place a shipped file gets read. features.py
# and train.py are scanned too, in case a future task references a filename
# directly from either of those instead (e.g. a feature builder that reads a
# CSV on its own). Read-only: this module never imports pipeline.evaluate,
# pipeline.submit, or anything else out of scope for B13.
_SCANNED_MODULE_NAMES: tuple[str, ...] = ("pipeline.data", "pipeline.features", "pipeline.train")


def _scanned_source() -> str:
    """Concatenated source text of every module in `_SCANNED_MODULE_NAMES`.

    Imports lazily and by name so a broken sibling module (B10/B11 features
    work, for instance) surfaces as a clear ImportError rather than this
    module silently not scanning it.
    """
    import importlib

    chunks = []
    for module_name in _SCANNED_MODULE_NAMES:
        module = importlib.import_module(module_name)
        chunks.append(inspect.getsource(module))
    return "\n".join(chunks)


# --- The exclusion map (B13 acceptance criteria) -----------------------------
#
# {shipped filename: reason}. Every shipped file not currently joined into
# the pipeline (per the static source-scan above) MUST have an entry here or
# check_data_usage() fails. When a future task wires one of these files into
# the pipeline, delete its entry here — the source scan will then correctly
# report it as "joined" instead.
EXCLUDED: dict[str, str] = {
    "user_features_pure.csv": (
        "Static per-user demographic/activity fields (user_active_degree, "
        "is_lowactive_period, is_live_streamer, is_video_author, "
        "follow/fans/friend counts and range buckets, register_days(_range), "
        "onehot_feat0..17) — all knowable before exposure, legitimately usable "
        "per the Section 6.8 feature policy's 'Static user' row. Not a leakage "
        "risk. It is simply not yet wired into the feature registry by any "
        "completed task — a candidate for a future feature task. Listed here "
        "so this checker stays green rather than silently ignoring an unused "
        "shipped file."
    ),
    "video_features_statistic_pure.csv": (
        "Aggregated per-video engagement-outcome counts (show/play/like/"
        "follow/share/complete-play counts and rates) accumulated over the "
        "*entire* collection window with no per-date breakdown at all. This "
        "is very likely exactly the 'monthly aggregate statistics file' "
        "Section 11 trap 3 warns about: its accumulation window may span the "
        "hidden-test period, so joining it would be leakage that 'looks "
        "completely innocuous'. Excluded by default per the Section 6.8 "
        "feature policy ('Monthly aggregate stats file' row) pending "
        "confirmed organiser guidance about a pre-split cutoff. NOTE: the "
        "literal string 'item_statistics_monthly' referenced by "
        "pipeline.features.EXCLUDED_SOURCES does not match this file's real "
        "shipped name (video_features_statistic_pure.csv) — a symbolic name "
        "in the plan's prose vs. the actual filename. Flagged explicitly "
        "here rather than silently worked around; pipeline.features.EXCLUDED_"
        "SOURCES may need a matching entry, but that file belongs to sibling "
        "B10/B11 tasks and is out of scope for this checker."
    ),
}


def check_data_usage(
    shipped_files: tuple[str, ...] = SHIPPED_FILES,
    excluded: dict[str, str] | None = None,
    source_text: str | None = None,
) -> list[str]:
    """Return the list of shipped files neither joined nor excluded.

    A shipped file is "joined" iff its literal filename string appears
    anywhere in the source of the scanned pipeline modules (see
    `_SCANNED_MODULE_NAMES`) — a static scan, not an attempt to actually
    load and run the pipeline. A file that is not joined must appear as a
    key in `excluded` (defaults to the module-level `EXCLUDED`).

    Empty result means every shipped file is accounted for. Non-empty names
    exactly the files that are neither joined nor excluded — a real gap:
    either wire the file into the pipeline, or add it to `EXCLUDED` with a
    reason.
    """
    excluded = EXCLUDED if excluded is None else excluded
    source_text = _scanned_source() if source_text is None else source_text

    unaccounted = []
    for filename in shipped_files:
        joined = filename in source_text
        if joined:
            continue
        if filename in excluded:
            continue
        unaccounted.append(filename)
    return unaccounted


def render_report(
    shipped_files: tuple[str, ...] = SHIPPED_FILES,
    excluded: dict[str, str] | None = None,
    unaccounted: list[str] | None = None,
) -> str:
    """Judge-visible run-log rendering: the shipped-file manifest, which
    files are excluded and why, and the outcome. Printed to stdout by the
    CLI entry point; also usable directly by tests or by a future D7/agent-
    loop caller that wants to capture this text into its own log.
    """
    excluded = EXCLUDED if excluded is None else excluded
    lines = ["=== B13 data-usage checker ===", "", "Shipped files:"]
    for filename in shipped_files:
        if filename in excluded:
            status = "EXCLUDED"
        elif unaccounted is not None and filename in unaccounted:
            status = "UNACCOUNTED FOR"
        else:
            status = "joined"
        lines.append(f"  [{status:16s}] {filename}")

    lines.append("")
    lines.append("Exclusion map (EXCLUDED = {filename: reason}):")
    if excluded:
        for filename, reason in excluded.items():
            lines.append(f"  - {filename}:")
            lines.append(f"      {reason}")
    else:
        lines.append("  (empty)")

    lines.append("")
    if unaccounted:
        lines.append(
            f"RESULT: FAIL — {len(unaccounted)} shipped file(s) neither joined "
            "nor excluded:"
        )
        for filename in unaccounted:
            lines.append(f"  - {filename}")
    else:
        lines.append("RESULT: OK — every shipped file is joined or explicitly excluded.")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    unaccounted = check_data_usage()
    print(render_report(unaccounted=unaccounted))
    return 1 if unaccounted else 0


if __name__ == "__main__":
    sys.exit(main())
