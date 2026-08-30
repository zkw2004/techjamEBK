"""Shared test helpers.

`main` stays green (Section 1.3). Unimplemented acceptance criteria are
recorded as skipped tests carrying their task ID, not as failures — so the
suite is a live checklist of what is left rather than a wall of red.

This file also owns **native-backend isolation**. torch and LightGBM each
link their own OpenMP runtime; a process that loads both aborts outright
with OMP Error #15 (see `pipeline/models/__init__.py`). `run_experiment`
sidesteps this by forking per experiment, but unit tests that drive a model
class directly would co-load both runtimes into the pytest process and kill
the whole session — which is exactly what used to happen the moment
`test_deepfm` (torch) ran before `test_lgbm` (LightGBM).

Tests that touch a native runtime therefore carry
`@pytest.mark.native_backend("torch" | "lightgbm")` and are executed in a
forked child, so the pytest process itself never commits to either runtime.
Linux shares one libgomp and would survive without this, but the isolation
is cheap and keeps the suite honest on the machines the demo runs on.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import sys
import traceback

import pytest

FORK_AVAILABLE = "fork" in mp.get_all_start_methods()


def todo(task: str):
    """Mark a test as awaiting its task. Remove the decorator when you build it."""
    return pytest.mark.skip(reason=f"awaiting task {task}")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "native_backend(name): run this test in a forked child so torch and "
        "LightGBM never co-load into the pytest process (OMP Error #15).",
    )


def _run_forked(func, kwargs) -> None:
    """Run one test body in a forked child; re-raise its failure here.

    The child inherits already-prepared fixtures through the fork, so the
    test body is unchanged. Only the outcome crosses the pipe: a formatted
    traceback for an ordinary failure, or the exit signal when the child
    dies natively (an abort produces no traceback at all).
    """
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(read_fd)
        payload = b""
        try:
            func(**kwargs)
        except BaseException as exc:  # noqa: BLE001 — reported, not handled
            payload = pickle.dumps(
                {
                    "failed": True,
                    "skipped": isinstance(exc, pytest.skip.Exception),
                    "reason": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        try:
            with os.fdopen(write_fd, "wb") as handle:
                handle.write(payload)
        finally:
            os._exit(0)

    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as handle:
        payload = handle.read()
    _, status = os.waitpid(pid, 0)

    if os.WIFSIGNALED(status):
        signal_number = os.WTERMSIG(status)
        pytest.fail(
            f"test body killed by signal {signal_number} in its isolated child. "
            "A native crash here is usually two OpenMP runtimes co-loaded "
            "(torch + LightGBM); check the native_backend marker and the "
            "model's declaration.",
            pytrace=False,
        )
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) != 0:
        pytest.fail(
            f"isolated child exited with code {os.WEXITSTATUS(status)}", pytrace=False
        )
    if not payload:
        return
    outcome = pickle.loads(payload)
    if outcome.get("skipped"):
        pytest.skip(outcome["reason"])
    pytest.fail(outcome["traceback"], pytrace=False)


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Divert `native_backend`-marked tests into a forked child."""
    marker = pyfuncitem.get_closest_marker("native_backend")
    if marker is None or not FORK_AVAILABLE:
        return None
    if sys.platform not in ("darwin", "linux"):
        return None
    kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    _run_forked(pyfuncitem.obj, kwargs)
    return True  # the hook handled the call
