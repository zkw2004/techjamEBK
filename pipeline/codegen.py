"""C4b: generated-feature vertical slice — the agent extends its own search space.

Owner: Workstream C (Ethan). Additive module; no frozen contract changes.

A typed config can only select features a human already wrote; Optuna can
search that menu without an LLM. Generated feature code is the one action an
LLM adds that parameter search structurally cannot: a candidate that did not
exist in the menu. It is also the most dangerous action — generated code can
trivially leak — so every generated feature runs the same gauntlet:

    syntax -> schema -> leakage audit -> smoke -> screen -> full
                                   |
                     accept, reject, or QUARANTINE

Quarantine is a status, not a deletion: the feature is deregistered so no
experiment can use it, and the append-only event ledger keeps the lineage
(Section 7.2 — the agent cannot erase its own failed experiment).

Generated features use the frozen Section 8.4 signature
``fn(train_df, target_df) -> np.ndarray`` and carry their emitted source on
``fn.__leak_source__`` so ``pipeline.features.leakage_check`` can audit code
that has no file behind it.
"""

from __future__ import annotations

import inspect
import time

from pipeline.features import FEATURES, leakage_check

GENERATED_PREFIX = "gen_"

# Vetting budget per tier. Smoke must stay interactive; full is the C1 default.
STAGE_TIMEOUTS_S = {"smoke": 60, "screen": 900, "full": 1800}

# --- The C4b deliverable feature: user-author affinity ----------------------
#
# Hypothesis: a user who long-viewed an author's videos before is likelier to
# long-view that author again. Computed with a strict PER-ROW temporal cutoff:
# each target row sees only training rows dated STRICTLY EARLIER than its own
# date (merge_asof with allow_exact_matches=False), so same-day and future
# interactions are invisible. Empirical-Bayes smoothing (alpha=20, Appendix
# A.1) keeps sparse user-author pairs near the global rate; fitting alpha on
# internal folds is B5's job and can replace the constant later.
USER_AUTHOR_AFFINITY_SOURCE = '''
def user_author_affinity(train_df, target_df):
    import numpy as np
    import pandas as pd

    if not hasattr(train_df, "groupby"):
        train_df = pd.DataFrame(dict(train_df))
    if not hasattr(target_df, "groupby"):
        target_df = pd.DataFrame(dict(target_df))

    label = "long_view"
    alpha = 20.0
    labels = pd.to_numeric(train_df[label], errors="coerce").fillna(0.0)
    global_rate = float(labels.mean()) if len(labels) else 0.0

    history = pd.DataFrame({
        "user_id": train_df["user_id"].astype(str).to_numpy(),
        "author_id": train_df["author_id"].astype(str).to_numpy(),
        "date_num": pd.to_numeric(train_df["date"], errors="coerce").to_numpy(),
        "outcome": labels.to_numpy(dtype=float),
    })
    grouped = (
        history.groupby(["user_id", "author_id", "date_num"], sort=False)["outcome"]
        .agg(["count", "sum"]).reset_index().sort_values("date_num", kind="stable")
    )
    cumulative = grouped.groupby(["user_id", "author_id"], sort=False)[["count", "sum"]].cumsum()
    grouped["seen"] = cumulative["count"]
    grouped["viewed"] = cumulative["sum"]

    target = pd.DataFrame({
        "user_id": target_df["user_id"].astype(str).to_numpy(),
        "author_id": target_df["author_id"].astype(str).to_numpy(),
        "date_num": pd.to_numeric(target_df["date"], errors="coerce").to_numpy(),
    })
    target["row"] = np.arange(len(target))

    merged = pd.merge_asof(
        target.sort_values("date_num", kind="stable"),
        grouped[["user_id", "author_id", "date_num", "seen", "viewed"]]
        .sort_values("date_num", kind="stable"),
        on="date_num",
        by=["user_id", "author_id"],
        allow_exact_matches=False,   # strictly earlier dates only
        direction="backward",
    ).sort_values("row", kind="stable")

    seen = merged["seen"].fillna(0.0).to_numpy(dtype=float)
    viewed = merged["viewed"].fillna(0.0).to_numpy(dtype=float)
    return (viewed + alpha * global_rate) / (seen + alpha)
'''

# --- The deliberately leaky twin (TEST FIXTURE ONLY) ------------------------
#
# Same shape, but it blends the target row's own outcome into the score. The
# column name is assembled at runtime, so a source grep for
# target_df["long_view"] does NOT catch it — that is the point: this twin
# exists to prove the DYNAMIC guard (leakage_check probe 3, which permutes
# target-side outcomes and watches the output) catches what a static scan
# cannot. It must never be registered outside a vet run, and vetting
# quarantines it.
LEAKY_TWIN_SOURCE = '''
def user_author_affinity_leaky(train_df, target_df):
    import numpy as np
    import pandas as pd

    if not hasattr(train_df, "groupby"):
        train_df = pd.DataFrame(dict(train_df))
    if not hasattr(target_df, "groupby"):
        target_df = pd.DataFrame(dict(target_df))

    label = "long" + "_view"   # evades a static grep for the literal column read
    labels = pd.to_numeric(train_df[label], errors="coerce").fillna(0.0)
    global_rate = float(labels.mean()) if len(labels) else 0.0
    rates = train_df.assign(_y=labels).groupby("author_id")["_y"].mean()
    base = target_df["author_id"].map(rates).fillna(global_rate).to_numpy(dtype=float)
    answer = pd.to_numeric(target_df[label], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return 0.5 * base + 0.5 * answer
'''


class GeneratedFeatureError(ValueError):
    """A generated feature failed validation before any training ran."""


def validate_syntax(code: str) -> None:
    """Raises SyntaxError with line context when the emitted code cannot parse."""
    compile(code, "<generated-feature>", "exec")


def load_feature(code: str, name: str | None = None):
    """Exec the emitted source and return the feature callable.

    The function must use the frozen Section 8.4 signature: exactly two
    positional parameters (train_df, target_df). The source is attached as
    ``__leak_source__`` so the leakage guard can audit it.
    """
    validate_syntax(code)
    namespace: dict = {}
    exec(code, namespace)  # noqa: S102 — isolated further by C1's subprocess at run time
    functions = {
        key: value
        for key, value in namespace.items()
        if callable(value) and not key.startswith("_") and inspect.isfunction(value)
    }
    if name is not None:
        if name not in functions:
            raise GeneratedFeatureError(
                f"emitted code defines no function named {name!r}; found {sorted(functions)}"
            )
        fn = functions[name]
    elif len(functions) == 1:
        fn = next(iter(functions.values()))
    else:
        raise GeneratedFeatureError(
            f"emitted code must define exactly one top-level function, found {sorted(functions)}"
        )

    parameters = [
        p for p in inspect.signature(fn).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(parameters) != 2:
        raise GeneratedFeatureError(
            f"generated feature {fn.__name__!r} must take exactly (train_df, target_df); "
            f"got {[p.name for p in parameters]}"
        )
    fn.__leak_source__ = code
    return fn


def registered_name(name: str) -> str:
    return name if name.startswith(GENERATED_PREFIX) else f"{GENERATED_PREFIX}{name}"


def register_generated(name: str, fn) -> str:
    """Register under the gen_ namespace. Replacing a gen_ entry is allowed
    (re-vetting), but a generated feature can never shadow a human-written one."""
    full_name = registered_name(name)
    existing = FEATURES.get(full_name)
    if existing is not None and not hasattr(existing, "__leak_source__"):
        raise GeneratedFeatureError(
            f"{full_name!r} is a human-registered feature; generated code cannot replace it"
        )
    FEATURES[full_name] = fn
    return full_name


def unregister_generated(name: str) -> None:
    full_name = registered_name(name)
    if hasattr(FEATURES.get(full_name), "__leak_source__"):
        del FEATURES[full_name]


def _stage(name: str, passed: bool, detail: str, seconds: float) -> dict:
    return {"stage": name, "passed": bool(passed), "detail": detail, "seconds": round(seconds, 3)}


def _trim(result: dict) -> dict:
    """Result evidence without the megabyte score arrays."""
    keep = ("status", "gauc", "ndcg", "primary", "fold_primaries", "seconds")
    return {key: result[key] for key in keep if key in result}


def _log_event(enabled: bool, event: dict) -> None:
    if not enabled:
        return
    from agent import store

    store.append_event(event)


def vet_generated_feature(
    name: str,
    code: str,
    base_config: dict | None = None,
    seed: int = 42,
    log_events: bool = True,
) -> dict:
    """Run one generated feature through the full containment gauntlet.

    Returns a report::

        {"feature": name, "registered_name": str | None,
         "status": "accepted" | "rejected" | "quarantined",
         "reason": str, "stages": [...], "results": {fidelity: trimmed}}

    * ``rejected``    — the code is broken (syntax, signature, runtime error);
      safe to repair and resubmit.
    * ``quarantined`` — the code LEAKS (static scan, dynamic probe, or the
      >0.75 canary). It is deregistered and must not be repaired by loosening
      the guard.
    * ``accepted``    — every stage passed; the feature stays registered and
      is usable in experiment configs under its ``gen_``-prefixed name.
    """
    from pipeline import train

    stages: list[dict] = []
    results: dict[str, dict] = {}

    def report(status: str, reason: str) -> dict:
        if status != "accepted":
            unregister_generated(name)
        _log_event(
            log_events,
            {
                "event": f"generated_feature_{status}",
                "feature": registered_name(name),
                "reason": reason,
                "stages": [item["stage"] for item in stages],
            },
        )
        return {
            "feature": name,
            "registered_name": registered_name(name) if status == "accepted" else None,
            "status": status,
            "reason": reason,
            "stages": stages,
            "results": results,
        }

    started = time.monotonic()
    try:
        validate_syntax(code)
    except SyntaxError as exc:
        detail = f"line {exc.lineno}: {exc.msg}"
        stages.append(_stage("syntax", False, detail, time.monotonic() - started))
        return report("rejected", f"syntax error in emitted code ({detail})")
    stages.append(_stage("syntax", True, "emitted code parses", time.monotonic() - started))

    started = time.monotonic()
    try:
        fn = load_feature(code)
    except GeneratedFeatureError as exc:
        stages.append(_stage("schema", False, str(exc), time.monotonic() - started))
        return report("rejected", str(exc))
    full_name = register_generated(name, fn)
    stages.append(
        _stage(
            "schema", True,
            f"signature matches Section 8.4; registered as {full_name!r}",
            time.monotonic() - started,
        )
    )

    started = time.monotonic()
    try:
        train_frame, validation_frame, _ = train._load_data()
        leakage_check(fn, train_frame, validation_frame)
    except (AssertionError, ValueError) as exc:
        stages.append(_stage("leakage", False, str(exc), time.monotonic() - started))
        return report("quarantined", f"leakage audit failed: {exc}")
    except Exception as exc:  # broken code discovered by the probes, not a leak
        detail = f"{type(exc).__name__}: {exc}"
        stages.append(_stage("leakage", False, detail, time.monotonic() - started))
        return report("rejected", f"feature raised during the leakage audit ({detail})")
    stages.append(
        _stage(
            "leakage", True,
            "static scan and dynamic outcome probes found no target-side reads",
            time.monotonic() - started,
        )
    )

    config = dict(base_config or {"model": "random"})
    features = [item for item in config.get("features", ["user_id", "video_id"])]
    if full_name not in features:
        features.append(full_name)
    config["features"] = features

    for fidelity in ("smoke", "screen", "full"):
        started = time.monotonic()
        result = train.run_experiment(
            config, fidelity=fidelity, seed=seed, timeout_s=STAGE_TIMEOUTS_S[fidelity]
        )
        results[fidelity] = _trim(result)
        if result["status"] != "ok":
            trace = str(result.get("traceback", "")).strip().splitlines()
            detail = (
                f"error_class={result.get('error_class')!r} at stage "
                f"{result.get('stage')!r}: {trace[-1] if trace else 'no traceback'}"
            )
            stages.append(_stage(fidelity, False, detail, time.monotonic() - started))
            if result.get("error_class") == "leak_suspected":
                return report(
                    "quarantined",
                    f"{fidelity} run tripped the leak canary "
                    f"(primary {result.get('primary')}); an implausibly strong score "
                    "is treated as cheating, not success",
                )
            return report("rejected", f"{fidelity} run failed: {detail}")
        stages.append(
            _stage(
                fidelity, True,
                f"{fidelity} run ok"
                + (
                    f" (primary {result['primary']:.4f})"
                    if result.get("primary") is not None
                    else ""
                ),
                time.monotonic() - started,
            )
        )

    return report("accepted", "passed syntax, schema, leakage, smoke, screen, and full gates")
