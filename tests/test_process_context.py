"""macOS multiprocessing safety for native ML workers."""

from __future__ import annotations

import sys

from agent import execute
from pipeline import train


def test_train_uses_spawn_after_torch_is_loaded_on_macos(monkeypatch):
    monkeypatch.delenv("TECHJAM_TEST_FORK_FIXTURES", raising=False)
    monkeypatch.setattr(train.sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "torch", object())

    assert train._process_context().get_start_method() == "spawn"


def test_execute_uses_spawn_after_torch_is_loaded_on_macos(monkeypatch):
    monkeypatch.delenv("TECHJAM_TEST_FORK_FIXTURES", raising=False)
    monkeypatch.setattr(execute.sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "torch", object())

    assert execute._process_context().get_start_method() == "spawn"
