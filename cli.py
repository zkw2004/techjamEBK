"""Local operational entry point for the requirements-first research loop."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

# Load .env before anything else touches os.environ. Without this, a real
# `run`/`selfcheck`/`baseline` in a fresh shell that never sourced .env fails
# every propose() call with an auth error and returns almost instantly, with
# zero real nodes written — the failure looks like nothing happened rather
# than like a config problem. override=False: real exported env vars (CI,
# a teammate's shell) still win over .env.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=False)


def _archive_current_run() -> Path | None:
    """Move prior evidence aside; fresh runs never delete the audit trail."""
    root = Path("logs")
    candidates = (root / "nodes", root / "run.jsonl", root / "manifest.json")
    if not any(path.exists() for path in candidates):
        return None
    archive = root / "archive" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive.mkdir(parents=True, exist_ok=False)
    for path in candidates:
        if path.exists():
            shutil.move(str(path), str(archive / path.name))
    return archive


def _install_fake_provider() -> None:
    """Install a deterministic proposal source used only by tests/smoke runs."""
    from agent import propose
    from agent.schema import Action

    def fake(history, _knowledge, parent):
        index = len(history) + 1
        return (
            Action(
                hypothesis=f"Deterministic smoke candidate {index} validates orchestration.",
                reasoning="Fake provider selected for deterministic test execution.",
                type="config",
                family="model",
                parent=parent["id"],
                config={"model": "random"},
            ),
            {"in": 0, "out": 0, "model": "deterministic-fake"},
        )

    propose.propose = fake  # type: ignore[assignment]


def _selfcheck(seed: int) -> int:
    from pipeline.train import run_experiment

    # run_experiment(fidelity="full") scores the official VALIDATION split,
    # never hidden test. K15/K16 (02_REQUIREMENTS.md) split the two: 0.4753 /
    # 0.5715 are the hidden-test rungs, 0.4834 / 0.5807 are validation. Using
    # the test-split numbers here compared apples to oranges and made this
    # gate fail on a correct run — verified 2026-08-31 against 5-seed
    # long_view measurements on the shipped starter kit.
    expected = {"random": 0.4834, "popularity": 0.5807}
    failures = []
    for model, target in expected.items():
        result = run_experiment({"model": model}, fidelity="full", seed=seed)
        primary = result.get("primary")
        ok = (
            result.get("status") == "ok"
            and isinstance(primary, (int, float))
            and abs(primary - target) < 0.0024
        )
        print(f"{model}: {primary!r}; expected {target:.4f}; {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(model)
    return 1 if failures else 0


def _baseline(seed: int) -> int:
    from pipeline.data import FIELDS
    from pipeline.train import run_experiment

    # Config.features defaults to ["user_id", "video_id"] (agent/schema.py) —
    # the official baseline needs all five FIELDS. Passing hparams.n_fields=5
    # alone is a no-op: FM.__init__ never reads an n_fields hparam, so the
    # missing "features" key silently trained on two fields and scored near
    # the popularity baseline (~0.58) instead of ~0.6016.
    result = run_experiment(
        {"model": "fm", "features": list(FIELDS), "hparams": {"k": 16, "lr": 0.001}},
        fidelity="full",
        seed=seed,
    )
    print(json.dumps(result, default=str, indent=2))
    primary = result.get("primary")
    ok = (
        result.get("status") == "ok"
        and isinstance(primary, (int, float))
        and abs(primary - 0.6016) < 0.0024
    )
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="run the independent D13 leakage canaries and exit",
    )
    commands = parser.add_subparsers(dest="command")
    for name in ("selfcheck", "baseline"):
        command = commands.add_parser(name)
        command.add_argument("--seed", type=int, default=42)
    probe = commands.add_parser("probes")
    probe.add_argument("--fidelity", choices=("screen", "full"), default="screen")
    probe.add_argument("--fm-trials", type=int, default=30)
    run = commands.add_parser("run")
    run.add_argument("--max-iterations", type=int, default=50)
    run.add_argument("--max-hours", type=float, default=6.0)
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--fresh", action="store_true", default=True)
    mode.add_argument("--resume", action="store_true")
    run.add_argument("--fake-provider", action="store_true")
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--node")
    finalize.add_argument("--output", type=Path, default=Path("submission.csv"))
    report = commands.add_parser("report")
    report.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.preflight:
        if args.command is not None:
            parser.error("--preflight cannot be combined with a command")
        from tools.leak_preflight import record_preflight, run_preflight, write_result

        result = run_preflight()
        write_result(result)
        record_preflight(result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command is None:
        parser.error("a command or --preflight is required")

    if args.command == "selfcheck":
        return _selfcheck(args.seed)
    if args.command == "baseline":
        return _baseline(args.seed)
    if args.command == "probes":
        from tools.probes import main as probes_main

        probes_main(["--fidelity", args.fidelity, "--fm-trials", str(args.fm_trials)])
        return 0
    if args.command == "run":
        if args.fresh and not args.resume:
            _archive_current_run()
        if args.fake_provider:
            _install_fake_provider()
        from agent import loop

        loop.run(max_iterations=args.max_iterations, max_hours=args.max_hours)
        print(loop.LAST_STOP_REASON or "iteration_cap")
        return 0
    if args.command == "finalize":
        from tools.finalise import finalise, selected_config

        result = finalise(selected_config(args.node), output_path=args.output)
        print(f"Wrote {result.output_path} ({result.row_count:,} rows, {len(result.seeds)} seeds)")
        return 0
    from tools.report import build_report, load_nodes, render_markdown

    payload = build_report(load_nodes(Path("logs/nodes")))
    print(json.dumps(payload, indent=2) if args.json else render_markdown(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
