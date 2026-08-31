"""Give every registered feature a fair test against the corrected FM encoder.

Until `pipeline/models/fm.py` gained a numeric path, FM built an exact-value
vocabulary over every column, so a continuous feature was memorised
value-by-value: `pcr_hist` produced ~1.0M distinct values over 1.14M training
rows with **100%** of validation rows unseen, making it a constant at scoring
time after polluting training with a million single-observation embeddings.
Thirteen of the ~20 registered features were affected. Every feature
experiment run against FM before that fix was therefore measuring the encoder,
not the feature -- see trap 14.

This sweeps the registry one feature at a time on top of the five official
fields and applies D12's combined rule to each: point delta >= MIN_DELTA_FLOOR
**and** a bootstrap 95% CI over users excluding zero.

Full tier is a single seed, so a survivor here is a *candidate*, not a result.
Anything that clears both conditions must be re-run at confirm tier (5 seeds)
before it is believed -- `--confirm` does exactly that for the survivors.

Note on the per-user aggregates. `tools/screen.py` measured eight features as
`metric_inert`: mean within-user variance exactly 0, because they are
`groupby("user_id")` aggregates and so take one identical value across every
candidate a user sees. Binning does not change that -- they still cannot move a
within-user metric directly. They are swept anyway because FM's second-order
term can still pair them with a video-varying field, and "expected to do
nothing" is a hypothesis worth one cheap measurement rather than an assumption.

Run:
    python3 scripts/feature_sweep.py                 # full tier, all features
    python3 scripts/feature_sweep.py --confirm       # confirm-tier the survivors
    python3 scripts/feature_sweep.py --features a b  # just these

The __main__ guard is mandatory: run_experiment isolates each fit in a spawned
subprocess and macOS spawn re-imports __main__ by path.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

RESULTS_PATH = Path("logs/feature_sweep.json")
BASELINE_HPARAMS = {"k": 16, "lr": 0.001}


def _run(features: list[str], fidelity: str, seed: int) -> dict:
    from pipeline.train import run_experiment

    started = time.time()
    result = run_experiment(
        {"model": "fm", "features": features, "hparams": dict(BASELINE_HPARAMS)},
        fidelity=fidelity,
        seed=seed,
    )
    result["_wall_s"] = time.time() - started
    return result


def _verdict(delta: float, ci: tuple[float, float], floor: float) -> str:
    """D12's combined rule, with the two failure modes kept distinct.

    A CI straddling zero means "indistinguishable from noise"; a CI entirely
    below zero means "confidently harmful", which is a different instruction --
    and, per trap 14, usually a signal to inspect the pipeline rather than to
    believe the number.
    """
    if delta >= floor and ci[0] > 0:
        return "CANDIDATE"
    if ci[1] < 0:
        return "harmful"
    if ci[0] > 0:
        return "positive-but-small"
    return "noise"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true",
                        help="re-run at confirm tier (5 seeds); by default only "
                             "survivors of a previous full-tier sweep")
    parser.add_argument("--features", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from agent import gate
    from pipeline.data import FIELDS
    from pipeline.features import FEATURES

    fidelity = "confirm" if args.confirm else "full"
    names = args.features
    if names is None and args.confirm and RESULTS_PATH.is_file():
        prior = json.loads(RESULTS_PATH.read_text())
        names = [r["feature"] for r in prior["results"]
                 if r["verdict"] in ("CANDIDATE", "positive-but-small")]
        if not names:
            print("No survivors from the full-tier sweep; nothing to confirm.")
            return 0
        print(f"Confirming {len(names)} survivor(s) from the full-tier sweep.\n")
    if names is None:
        names = sorted(FEATURES)

    print(f"Feature sweep @ {fidelity} tier, seed {args.seed}, "
          f"floor {gate.MIN_DELTA_FLOOR}")
    print(f"baseline: fm + {len(FIELDS)} official fields\n")

    base = _run(list(FIELDS), fidelity, args.seed)
    if base["status"] != "ok":
        print(f"baseline FAILED: {base.get('error_class')} {base.get('error')}")
        return 1
    base_primary = float(base["primary"])
    print(f"{'baseline':34s} {base_primary:.6f}  [{base['_wall_s']:.0f}s]\n")

    print(f"{'feature':34s} {'primary':>9s} {'delta':>10s} "
          f"{'95% CI':>24s}  verdict")
    print("-" * 100)

    results = []
    for name in names:
        run = _run([*FIELDS, name], fidelity, args.seed)
        if run["status"] != "ok":
            print(f"{name:34s} {'ERROR':>9s} {run.get('error_class', ''):>10s}  "
                  f"{str(run.get('error'))[:40]}")
            results.append({"feature": name, "verdict": "error",
                            "error": str(run.get("error"))[:400]})
            continue
        primary = float(run["primary"])
        delta = primary - base_primary
        _, ci = gate.accept(run["val_scores"], base["val_scores"], base["val_user_ids"])
        verdict = _verdict(delta, ci, gate.MIN_DELTA_FLOOR)
        results.append({"feature": name, "primary": primary, "delta": delta,
                        "ci": list(ci), "verdict": verdict,
                        "wall_s": run["_wall_s"]})
        print(f"{name:34s} {primary:9.6f} {delta:+10.6f} "
              f"[{ci[0]:+.6f}, {ci[1]:+.6f}]  {verdict}", flush=True)

    results.sort(key=lambda r: r.get("delta", -9), reverse=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(
        {"fidelity": fidelity, "seed": args.seed, "baseline_primary": base_primary,
         "min_delta_floor": gate.MIN_DELTA_FLOOR, "results": results}, indent=2))

    candidates = [r for r in results if r.get("verdict") == "CANDIDATE"]
    errored = [r for r in results if r.get("verdict") == "error"]
    print("\n" + "=" * 100)
    if errored:
        # A sweep that lost runs has not tested the registry and must not report
        # as though it had. "No feature cleared the floor" reads as a finding
        # about the features; with runs missing it is a finding about the sweep.
        # This is not hypothetical -- the first full run of this script lost 15
        # of 21 features to a FileNotFoundError (the working-tree copy of this
        # file was removed by a branch switch mid-run, and macOS spawn
        # re-imports __main__ by path in every child) and still printed a
        # confident "no feature clears the floor". Name the gap; exit non-zero.
        print(f"INCOMPLETE: {len(errored)} of {len(results)} feature(s) errored "
              f"and were NOT tested:")
        for item in errored:
            print(f"  {item['feature']}")
        print("\nNo conclusion is drawn from a partial sweep. Re-run the missing")
        print("features before reading anything into the ones that finished:")
        print("  python3 scripts/feature_sweep.py --features "
              + " ".join(item["feature"] for item in errored))
        print(f"\nPartial results written to {RESULTS_PATH}")
        return 1
    if candidates:
        print(f"{len(candidates)} feature(s) clear D12's combined rule at "
              f"{fidelity} tier:")
        for r in candidates:
            print(f"  {r['feature']:32s} {r['delta']:+.6f}  "
                  f"CI [{r['ci'][0]:+.6f}, {r['ci'][1]:+.6f}]")
        if fidelity == "full":
            print("\nSingle seed -- these are CANDIDATES, not results. Re-run:")
            print("  python3 scripts/feature_sweep.py --confirm")
    else:
        print(f"No feature clears the floor at {fidelity} tier.")
        print("That is a reportable result: with the encoder corrected, the")
        print("registry's single-feature additions do not beat the five")
        print("official fields. Combinations remain untested.")
    print(f"\nWritten to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
