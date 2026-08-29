"""Shared test helpers.

`main` stays green (Section 1.3). Unimplemented acceptance criteria are
recorded as skipped tests carrying their task ID, not as failures — so the
suite is a live checklist of what is left rather than a wall of red.
"""

from __future__ import annotations

import pytest


def todo(task: str):
    """Mark a test as awaiting its task. Remove the decorator when you build it."""
    return pytest.mark.skip(reason=f"awaiting task {task}")
