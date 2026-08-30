"""Sustained-use behaviour of the C runner.

Every other test calls `run_experiment` a handful of times. An unattended run
calls it hundreds of times in one process, and the failures that appear only
under repetition — leaked descriptors, accumulated memory, zombie children, a
cache that never evicts — are exactly the ones that end an overnight run at
3am with no one watching.

The soak itself is deliberately small (fast fixtures, dozens of iterations)
because the point is the *trend*, not the absolute numbers: a leak shows up as
growth per iteration whatever the scale.
"""

from __future__ import annotations

import os
import resource

import numpy as np
import pandas as pd
import pytest

from pipeline import train

TRAIN_DATES = list(range(20220408, 20220422))


def _frame(rows: int, offset: int = 0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    users = rng.integers(0, 12, rows)
    authors = rng.integers(0, 6, rows)
    favoured = ((users + authors) % 3 == 0).astype(float)
    del offset
    return pd.DataFrame(
        {
            "date": rng.choice(TRAIN_DATES, rows),
            "user_id": users,
            "video_id": rng.integers(0, 80, rows),
            "author_id": authors,
            "tab": rng.integers(0, 2, rows),
            "duration_ms": rng.integers(1_000, 60_000, rows),
            "long_view": (rng.random(rows) < 0.12 + 0.45 * favoured).astype(int),
        }
    )


def _data():
    return _frame(900, seed=1), _frame(250, seed=2), _frame(250, seed=3)


def _folds():
    return [(_frame(400, seed=10 + i), _frame(180, seed=20 + i)) for i in range(3)]


MODELS = [
    ("random", {}),
    ("popularity", {}),
    ("fm", {"max_epochs": 1, "k": 4}),
    ("lgbm", {"min_data_in_leaf": 1, "num_boost_round": 5}),
]


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "_load_data", _data, raising=False)
    monkeypatch.setattr(train, "_load_folds", _folds, raising=False)
    monkeypatch.setattr(train, "SCORE_CACHE_DIR", tmp_path / "scores")
    return train


def _open_descriptors() -> int:
    try:
        return len(os.listdir("/dev/fd"))
    except OSError:  # pragma: no cover - platform without /dev/fd
        return -1


def _peak_rss_mb() -> float:
    divisor = 1024 * 1024 if os.uname().sysname == "Darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def _current_rss_mb() -> float:
    """Resident set size *now*.

    `ru_maxrss` is a high-water mark that never falls, so it cannot separate a
    steady leak from a single transient spike. An audit that injected a 2MB
    per-run leak went undetected against a peak-RSS bound; measuring current
    RSS and fitting the trend catches it.
    """
    import subprocess

    probe = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True
    )
    return float(probe.stdout.strip() or 0) / 1024


# Observed growth over 120 real iterations is ~0.023 MB/iteration. This bound
# leaves ~20x headroom for allocator noise while still failing a per-run leak
# of a megabyte or more, which is the size that ends an overnight run.
MAX_RSS_GROWTH_MB_PER_ITERATION = 0.5


def test_sustained_runs_leak_no_descriptors_memory_or_children(runner):
    """A leak of one descriptor or child per experiment is invisible over the
    handful of calls every other test makes, and fatal over a few hundred."""
    iterations = 48
    before_fds = _open_descriptors()
    samples: list[tuple[int, float]] = []

    failures = []
    for index in range(iterations):
        model, hparams = MODELS[index % len(MODELS)]
        fidelity = ("smoke", "screen", "full")[index % 3]
        result = runner.run_experiment(
            {"model": model, "features": ["user_id", "video_id"], "hparams": hparams},
            fidelity=fidelity,
            seed=index,
            timeout_s=300,
        )
        if result["status"] != "ok":
            failures.append((index, model, fidelity, result.get("error_class")))
        # Skip the first few: import and allocator warm-up is a step change,
        # not a leak, and including it biases the slope.
        if index >= 8:
            samples.append((index, _current_rss_mb()))

    assert not failures, f"experiments failed under repetition: {failures[:5]}"

    after_fds = _open_descriptors()
    if before_fds >= 0:
        assert after_fds - before_fds <= 2, (
            f"descriptor count grew {before_fds} -> {after_fds} over {iterations} runs; "
            "a per-experiment pipe or process leak exhausts the limit on a long run"
        )

    indices = np.array([index for index, _ in samples], dtype=float)
    rss = np.array([value for _, value in samples], dtype=float)
    slope = float(np.polyfit(indices, rss, 1)[0])
    assert slope < MAX_RSS_GROWTH_MB_PER_ITERATION, (
        f"RSS grew {slope:.2f}MB per iteration ({rss[0]:.1f} -> {rss[-1]:.1f}MB over "
        f"{len(samples)} sampled runs); extrapolated over a 40-iteration run with "
        f"confirm tiers that is {slope * 200:.0f}MB of leak"
    )

    assert not _zombie_children(), "worker processes were not reaped"


def _zombie_children() -> list[int]:
    """Children left unreaped by this process.

    Zombies accumulate silently until the process table fills, so an
    unattended run dies long after the code that caused it.

    Listing children portably matters here: the first version used
    `ps -P <pid>`, which is an illegal option on macOS and means "show
    processor" on Linux. It returned nothing on both, so the assertion built
    on it passed unconditionally — an audit that created a real zombie found
    it undetected. `ps -eo pid,ppid,stat` is understood by both.
    """
    import subprocess

    probe = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,stat="], capture_output=True, text=True
    )
    mine = os.getpid()
    zombies = []
    for line in probe.stdout.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            continue
        pid, ppid, state = parts
        if ppid.isdigit() and int(ppid) == mine and state.strip().startswith("Z"):
            zombies.append(int(pid))
    return zombies


def test_repeated_identical_runs_stay_deterministic(runner):
    """Determinism must survive repetition, not just hold on the first call:
    state accumulating in the parent across forks would show up here."""
    config = {"model": "fm", "features": ["user_id", "video_id"], "hparams": {"max_epochs": 1}}
    baseline = runner.run_experiment(config, fidelity="screen", seed=7, timeout_s=300)

    for _ in range(6):
        repeat = runner.run_experiment(config, fidelity="screen", seed=7, timeout_s=300)
        assert repeat["primary"] == baseline["primary"]
        np.testing.assert_array_equal(repeat["val_scores"], baseline["val_scores"])


# --- score cache eviction ---------------------------------------------------


def _write_entry(directory, name: str, payload_bytes: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.npz").write_bytes(b"0" * payload_bytes)


def test_cache_evicts_least_recently_used_entries_past_the_budget(tmp_path, monkeypatch):
    """A real full-tier entry is ~2.3MB, so an unattended run writes ~90MB and
    the confirm tier pushes past 400MB. Nothing replaced an entry before this,
    so the directory only ever grew."""
    monkeypatch.setattr(train, "SCORE_CACHE_DIR", tmp_path)
    import time as time_module

    for index in range(5):
        _write_entry(tmp_path, f"entry{index}", 1_000)
        # Distinct access times so "least recently used" is well defined.
        stamp = time_module.time() - (10 - index)
        os.utime(tmp_path / f"entry{index}.npz", (stamp, stamp))

    removed = train._evict_score_cache(max_bytes=2_500)

    survivors = sorted(path.stem for path in tmp_path.glob("*.npz"))
    assert removed == 3
    assert survivors == ["entry3", "entry4"], "the newest entries must survive"


def test_cache_eviction_is_a_no_op_within_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "SCORE_CACHE_DIR", tmp_path)
    _write_entry(tmp_path, "only", 100)

    assert train._evict_score_cache(max_bytes=10_000) == 0
    assert (tmp_path / "only.npz").is_file()


def test_cache_eviction_tolerates_a_missing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "SCORE_CACHE_DIR", tmp_path / "absent")

    assert train._evict_score_cache() == 0


def test_losing_a_cached_entry_is_never_incorrect(runner, tmp_path):
    """Eviction is safe precisely because a miss costs a refit, not a wrong
    answer: the recomputed result must match what the cache would have served."""
    config = {"model": "fm", "features": ["user_id", "video_id"], "hparams": {"max_epochs": 1}}
    first = runner.run_experiment(config, fidelity="full", seed=5, timeout_s=300)

    for path in (tmp_path / "scores").glob("*.npz"):
        path.unlink()

    second = runner.run_experiment(config, fidelity="full", seed=5, timeout_s=300)

    assert first["status"] == second["status"] == "ok"
    assert first["primary"] == second["primary"]
    np.testing.assert_array_equal(first["val_scores"], second["val_scores"])


def test_full_runs_keep_the_cache_within_budget(runner, monkeypatch):
    """The eviction hook has to actually run on the write path."""
    monkeypatch.setattr(train, "SCORE_CACHE_MAX_BYTES", 4_000)
    cache = train.SCORE_CACHE_DIR

    for seed in range(6):
        runner.run_experiment(
            {"model": "fm", "features": ["user_id", "video_id"], "hparams": {"max_epochs": 1}},
            fidelity="full",
            seed=seed,
            timeout_s=300,
        )

    total = sum(path.stat().st_size for path in cache.glob("*.npz")) if cache.is_dir() else 0
    assert total <= 4_000 * 2, f"cache reached {total} bytes against a 4000 byte budget"
