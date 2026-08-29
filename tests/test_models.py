"""C2/C3 acceptance: the reference rungs reproduce their published numbers.

C2 IS A HARD GATE — if random and popularity do not reproduce 0.4753 and
0.5715, nothing downstream means anything.
"""

from __future__ import annotations

from pipeline.models import MODEL_REGISTRY

from tests.conftest import todo

EXPECTED_MODELS = {"random", "popularity", "fm", "lgbm", "deepfm", "deepfm_mtl", "blend"}


def test_registry_covers_every_model_in_the_config_schema():
    from agent.schema import Config
    schema_models = set(Config.model_fields["model"].annotation.__args__)
    assert schema_models == EXPECTED_MODELS
    assert EXPECTED_MODELS <= set(MODEL_REGISTRY)


def test_unknown_model_lookup_raises_before_training():
    import pytest

    from pipeline.models import get
    with pytest.raises(KeyError):
        get("wide_and_deep")


@todo("C2")
def test_random_reproduces_reference_primary():
    """0.4753 within noise. GATE."""


@todo("C2")
def test_popularity_reproduces_reference_primary():
    """0.5715 within noise. GATE."""


@todo("C3")
def test_fm_reproduces_baseline_validation_primary():
    """0.6016 within one seed-std (0.0008)."""


@todo("C1")
def test_run_experiment_is_deterministic_given_seed():
    """Two same-seed runs must produce identical output."""


@todo("C1")
def test_run_experiment_never_raises():
    """Every failure path returns status="error" with a classified error_class."""


@todo("C1")
def test_smoke_tier_completes_under_ten_seconds():
    pass
