"""B14's decision gate: does `sim_to_history` earn C10 (DIN)?

AGENT_PLAN.md 9.2, row B14:

    Decision gate: FM + 5 official fields + sim_to_history, confirm-tier
    (5 seeds) vs. baseline. Clears MIN_DELTA_FLOOR with a CI excluding
    zero -> funds C10. Lands inside noise -> C10 is not built.

Both halves of D12's combined rule are evaluated here, because either one
alone is misleading: a point delta over the floor can still have a CI that
straddles zero (noise that happened to land high), and a CI excluding zero
can still sit under the floor (a real but too-small effect). The loop's
`_accept_full` requires both, so this gate does too.

The comparison is against a *freshly measured* FM run on the same five
fields, not against the published 0.6016 constant. Like-for-like matters:
the CI needs two aligned score vectors from the same code path, and a delta
against a number from a different codebase would confound the feature's
effect with every other difference between the two. The published constant
is still checked separately, as the absolute bar `_accept_full` gates on.

Run:  python3 scripts/b14_decision_gate.py [--fidelity smoke|full|confirm]

The __main__ guard is mandatory: run_experiment isolates each fit in a
spawned subprocess, and macOS spawn re-imports __main__ *by path*, so an
unguarded module body re-runs the whole sweep inside every child.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

BASELINE_HPARAMS = {"k": 16, "lr": 0.001}
CANDIDATE_FEATURE = "sim_to_history"


def _run(config: dict, fidelity: str, seed: int) -> dict:
    from pipeline.train import run_experiment

    started = time.time()
    result = run_experiment(config, fidelity=fidelity, seed=seed)
    result["_wall_s"] = time.time() - started
    return result


def _primary(result: dict) -> float | None:
    """Full/confirm report `primary`; screen leaves it None and fills folds."""
    if result.get("primary") is not None:
        return float(result["primary"])
    folds = result.get("fold_primaries") or []
    return float(np.mean(folds)) if folds else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fidelity", default="confirm",
                        choices=("smoke", "screen", "full", "confirm"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from agent import gate
    from agent.manifest import BASELINE_VALIDATION
    from pipeline.data import FIELDS

    baseline_config = {
        "model": "fm",
        "features": list(FIELDS),
        "hparams": dict(BASELINE_HPARAMS),
    }
    candidate_config = {
        "model": "fm",
        "features": [*FIELDS, CANDIDATE_FEATURE],
        "hparams": dict(BASELINE_HPARAMS),
    }

    print(f"B14 decision gate @ {args.fidelity} tier, seed {args.seed}")
    print(f"  baseline : fm + {len(FIELDS)} official fields")
    print(f"  candidate: fm + {len(FIELDS)} official fields + {CANDIDATE_FEATURE}\n")

    runs = {}
    for label, config in (("baseline", baseline_config), ("candidate", candidate_config)):
        result = _run(config, args.fidelity, args.seed)
        if result["status"] != "ok":
            print(f"{label:9s} FAILED [{result.get('error_class')}] "
                  f"{str(result.get('error'))[:400]}")
            return 1
        runs[label] = result
        value = _primary(result)
        shown = f"{value:.6f}" if value is not None else "n/a (smoke)"
        print(f"{label:9s} primary={shown}  [{result['_wall_s']:.0f}s]", flush=True)

    if args.fidelity == "smoke":
        print("\nSmoke tier only checks that both configs run; no metrics by design.")
        return 0

    base_primary = _primary(runs["baseline"])
    cand_primary = _primary(runs["candidate"])
    delta = cand_primary - base_primary

    print(f"\n{'':14s}{'primary':>10s}")
    print(f"{'baseline':14s}{base_primary:10.6f}")
    print(f"{'candidate':14s}{cand_primary:10.6f}")
    print(f"{'delta':14s}{delta:+10.6f}   floor {gate.MIN_DELTA_FLOOR}")

    # --- D12's combined rule -------------------------------------------------
    clears_floor = delta >= gate.MIN_DELTA_FLOOR
    ci_excludes_zero = ci = None
    if args.fidelity in ("full", "confirm"):
        ci_excludes_zero, ci = gate.accept(
            runs["candidate"]["val_scores"],
            runs["baseline"]["val_scores"],
            runs["baseline"]["val_user_ids"],
        )
        print(f"{'95% CI':14s}[{ci[0]:+.6f}, {ci[1]:+.6f}]  "
              f"excludes zero: {ci_excludes_zero}")

    beats_published = cand_primary >= BASELINE_VALIDATION + gate.MIN_DELTA_FLOOR
    print(f"\npublished baseline {BASELINE_VALIDATION:.4f}; candidate "
          f"{'clears' if beats_published else 'does NOT clear'} it by the floor")

    verdict = bool(clears_floor and ci_excludes_zero)
    print("\n" + "=" * 62)
    print(f"VERDICT: {'FUND C10 (DIN)' if verdict else 'DO NOT BUILD C10'}")
    print("=" * 62)
    if verdict:
        print("sim_to_history clears the floor with a CI excluding zero: the")
        print("user-history x candidate family is live. C10 is funded.")
    else:
        reasons = []
        if not clears_floor:
            reasons.append(f"point delta {delta:+.6f} < floor {gate.MIN_DELTA_FLOOR}")
        if ci is not None:
            # `gate.accept` returns `low > 0`, so a False verdict covers two
            # very different situations and they must not be reported as one.
            # A CI straddling zero means "indistinguishable from noise"; a CI
            # lying entirely below zero means "confidently harmful", which is a
            # much stronger claim and a different instruction to the team.
            if ci[1] < 0:
                reasons.append(
                    f"95% CI [{ci[0]:+.6f}, {ci[1]:+.6f}] lies entirely BELOW "
                    "zero -- the feature is confidently harmful, not merely "
                    "unproven"
                )
            elif ci[0] <= 0 <= ci[1]:
                reasons.append(
                    f"95% CI [{ci[0]:+.6f}, {ci[1]:+.6f}] straddles zero "
                    "-- indistinguishable from noise"
                )
        print("Gate not met: " + "; ".join(reasons) + ".")
        print("Per B14's row, C10 is NOT built. This is a real result, not a")
        print("failure -- it is a direct test of the organisers' #2 ranked")
        print("headroom direction at a fraction of DIN's cost.")
        if ci is not None and ci[1] < 0:
            print("\nNote: a confidently NEGATIVE result tests this *feature*,")
            print("not the whole user-history x candidate family. A feature")
            print("that actively hurts may be mis-specified rather than")
            print("uninformative -- check it is not adding a noisy field that")
            print("FM must spend embedding capacity on before concluding the")
            print("family is dead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
