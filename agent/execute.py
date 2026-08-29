"""execute(): Action -> node record. Task A4.

type="code" runs in an isolated subprocess: buggy-not-malicious code must
return status="error" with a traceback rather than crashing the parent, and
a hang must be killed at timeout. Subprocess isolation is sufficient here —
no Docker (Section 12).
"""

from __future__ import annotations

from agent.schema import Action


def execute(action: Action, fidelity: str = "smoke", timeout_s: int = 1800) -> dict:
    """Build the config from the action, call run_experiment, return a node dict."""
    raise NotImplementedError("A4")
