"""execute(): Action -> node record. Task A4.

type="code" runs in an isolated subprocess: buggy-not-malicious code must
return status="error" with a traceback rather than crashing the parent, and
a hang must be killed at timeout. Subprocess isolation is sufficient here —
no Docker (Section 12).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import traceback as traceback_module
from collections.abc import Callable
from numbers import Real
from typing import Any

from agent.schema import Action

PROCESS_STOP_GRACE_S = 0.1


def _run_experiment(
    config: dict, fidelity: str, seed: int, timeout_s: int | float
) -> dict:
    """Late import keeps this module importable while C1 is being developed."""
    from pipeline.train import run_experiment

    return run_experiment(config, fidelity=fidelity, seed=seed, timeout_s=timeout_s)


def _error_class(exc: BaseException) -> str:
    if isinstance(exc, SyntaxError):
        return "syntax"
    if isinstance(exc, MemoryError):
        return "oom"
    if isinstance(exc, (KeyError, TypeError, ValueError)):
        return "schema"
    return "transient"


def _execution_error(
    action: Action,
    fidelity: str,
    started: float,
    stage: str,
    error_class: str,
    traceback_text: str,
) -> dict:
    """Build a failed node without allowing executor failures to escape."""
    return _node(
        action,
        fidelity,
        {
            "status": "error",
            "stage": stage,
            "error_class": error_class,
            "traceback": traceback_text,
            "seconds": time.monotonic() - started,
        },
    )


def _diff(action: Action) -> str:
    """A compact, truthful diff summary for the append-only node ledger."""
    if action.type == "code":
        return "generated Python source executed in isolated subprocess"
    if action.type == "tune":
        return f"hyperparameter search: {sorted((action.search_space or {}).keys())}"
    if action.config is None:
        return ""
    return f"{action.type}: model={action.config.model}, loss={action.config.loss}"


def _node(action: Action, fidelity: str, result: dict, *, accepted: bool = False) -> dict:
    """Translate C1's public result contract into the Section 8.7 node shape.

    ``val_scores``/``val_user_ids`` live only on ``result`` (C1's contract,
    Section 8.5) — the Section 8.7 node schema has no field for them, so they
    never reach the persisted record. ``accepted`` must therefore be decided
    by the caller, on ``result``, before this conversion drops them; it is
    not computed here.
    """
    config = action.config.model_dump() if action.config is not None else {}
    node: dict[str, Any] = {
        "parent": action.parent,
        "family": action.family,
        "hypothesis": action.hypothesis,
        "reasoning": action.reasoning,
        "action_type": action.type,
        "fidelity": fidelity,
        "config": config,
        "diff": _diff(action),
        "status": result.get("status", "error"),
        "accepted": bool(accepted) if result.get("status") == "ok" else False,
        "manual_intervention": False,
        "seconds": result.get("seconds", 0.0),
        "gpu_seconds": result.get("gpu_seconds", 0.0),
    }

    if result.get("status") == "ok":
        metrics = {
            name: result[name]
            for name in ("gauc", "ndcg", "primary")
            if result.get(name) is not None
        }
        node.update(
            metrics=metrics,
            fold_primaries=result.get("fold_primaries", []),
            segments=result.get("segments", {}),
            errors=[],
        )
    else:
        node.update(
            metrics={},
            fold_primaries=[],
            segments={},
            errors=[
                {
                    "stage": result.get("stage", fidelity),
                    "error_class": result.get("error_class", "transient"),
                    "traceback": result.get("traceback", "executor returned an invalid error"),
                }
            ],
        )
    return node


def _process_context():
    """Use a clean child after PyTorch/MPS has touched macOS Objective-C state."""
    methods = mp.get_all_start_methods()
    if os.environ.get("TECHJAM_TEST_FORK_FIXTURES") == "1" and "fork" in methods:
        return mp.get_context("fork")
    if sys.platform == "darwin" and "spawn" in methods:
        return mp.get_context("spawn")
    return mp.get_context("fork" if "fork" in methods else "spawn")


def _stop_process(process: Any) -> None:
    if process is None or not process.is_alive():
        return
    process.terminate()
    process.join(PROCESS_STOP_GRACE_S)
    if process.is_alive():
        process.kill()
        process.join(PROCESS_STOP_GRACE_S)


def _code_child(
    send_conn: Any,
    code: str,
    config: dict,
    fidelity: str,
    seed: int,
    timeout_s: int | float,
    family: str = "model",
) -> None:
    """Run generated code and C1 in a child; send only a serialisable result.

    family="feature" routes through the C4b codegen path: the emitted source
    is loaded against the frozen Section 8.4 signature, registered under its
    gen_-prefixed name (in this child only — registration dies with the
    process), and appended to the config's feature list so the code actually
    participates in the run. C1's _matrix then applies the B6 leakage guard
    to it before any training, via the attached __leak_source__.
    """
    try:
        if family == "feature":
            from pipeline import codegen

            fn = codegen.load_feature(code)
            full_name = codegen.register_generated(fn.__name__, fn)
            features = list(config.get("features", []))
            if full_name not in features:
                features.append(full_name)
            config = {**config, "features": features}
        else:
            namespace = {"__name__": "generated_experiment"}
            exec(compile(code, "<generated-experiment>", "exec"), namespace)  # noqa: S102
        result = _run_experiment(config, fidelity, seed, timeout_s)
        send_conn.send(("result", result))
    except BaseException as exc:
        send_conn.send(
            (
                "error",
                {
                    "stage": "code",
                    "error_class": _error_class(exc),
                    "traceback": traceback_module.format_exc(),
                },
            )
        )
    finally:
        send_conn.close()


def _run_code_action(
    action: Action, config: dict, fidelity: str, timeout_s: int | float
) -> dict:
    """Execute code in an isolated process and enforce the outer timeout."""
    receive_conn = send_conn = process = None
    try:
        context = _process_context()
        receive_conn, send_conn = context.Pipe(duplex=False)
        # Deliberately not daemon=True: C1 creates its own worker process.
        process = context.Process(
            target=_code_child,
            args=(
                send_conn, action.code or "", config, fidelity,
                action.config.seed, timeout_s, action.family,
            ),
        )
        process.start()
        send_conn.close()
        send_conn = None

        if not receive_conn.poll(timeout_s):
            _stop_process(process)
            return {
                "status": "error",
                "stage": "code",
                "error_class": "timeout",
                "traceback": f"generated code exceeded timeout_s={timeout_s}",
            }
        try:
            kind, payload = receive_conn.recv()
        except (EOFError, OSError) as exc:
            return {
                "status": "error",
                "stage": "code",
                "error_class": "transient",
                "traceback": "".join(traceback_module.format_exception_only(type(exc), exc)),
            }
        finally:
            process.join(PROCESS_STOP_GRACE_S)
            _stop_process(process)

        if kind == "result" and isinstance(payload, dict):
            return payload
        if kind == "error" and isinstance(payload, dict):
            return {"status": "error", **payload}
        return {
            "status": "error",
            "stage": "code",
            "error_class": "schema",
            "traceback": "generated-code worker returned an invalid result",
        }
    finally:
        if receive_conn is not None:
            receive_conn.close()
        if send_conn is not None:
            send_conn.close()


def execute(
    action: Action,
    fidelity: str = "smoke",
    timeout_s: int = 1800,
    *,
    accept_fn: Callable[[dict], bool] | None = None,
) -> dict:
    """Build an Action into one completed node; never raise into the loop.

    C1 owns model execution and its own timeout.  A generated-code Action has
    an additional outer subprocess so untrusted feature/model source cannot
    crash or hang the orchestration parent.

    ``accept_fn``, when given, is called on the raw C1 result (which still
    carries ``val_scores``/``val_user_ids``) before it is converted to the
    Section 8.7 node shape, since that conversion drops those fields. Its
    return value becomes the persisted node's ``accepted`` field. Only called
    on a successful result — the node schema forces ``accepted=False`` on any
    error regardless.
    """
    started = time.monotonic()
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, Real) or timeout_s <= 0:
        return _execution_error(
            action,
            fidelity,
            started,
            "validation",
            "schema",
            "timeout_s must be a positive number",
        )
    if action.config is None:
        return _execution_error(
            action,
            fidelity,
            started,
            "validation",
            "schema",
            f"type={action.type!r} requires a concrete config before execution",
        )
    if action.type == "code" and not action.code:
        return _execution_error(
            action,
            fidelity,
            started,
            "validation",
            "schema",
            "type='code' requires Python source in action.code",
        )

    config = action.config.model_dump()
    try:
        if action.type == "code":
            result = _run_code_action(action, config, fidelity, timeout_s)
        else:
            result = _run_experiment(config, fidelity, action.config.seed, timeout_s)
        if not isinstance(result, dict):
            raise TypeError(f"run_experiment returned {type(result).__name__}, expected dict")
        # The node is the audited unit of work, so account for generated-code
        # setup and IPC as well as C1's model-training duration.
        result["seconds"] = time.monotonic() - started
        accepted = False
        if accept_fn is not None and result.get("status") == "ok":
            accepted = accept_fn(result)
        return _node(action, fidelity, result, accepted=accepted)
    except BaseException as exc:
        return _execution_error(
            action,
            fidelity,
            started,
            fidelity,
            _error_class(exc),
            traceback_module.format_exc(),
        )
