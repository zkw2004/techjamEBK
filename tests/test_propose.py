"""A3 acceptance: given a fake node history, returns a valid Action; token
usage is captured; two-tier routing (Opus propose, Haiku repair).

No network. The client is injected, which is why propose() takes one.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent import propose as P
from agent.schema import Action

VALID_ACTION = Action(
    hypothesis="lambdarank should beat pointwise BCE since GAUC is per-user",
    reasoning="n009 plateaued on fold 3",
    type="config",
    family="objective",
    parent="n009",
    config={"model": "lgbm", "loss": "lambdarank"},
)


class FakeMessages:
    """Records each call and returns queued responses."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        parsed, usage = self.outputs.pop(0)
        if isinstance(parsed, Exception):
            raise parsed
        return SimpleNamespace(parsed_output=parsed, usage=SimpleNamespace(**usage))


class FakeClient:
    def __init__(self, outputs):
        self.messages = FakeMessages(outputs)


USAGE = {
    "input_tokens": 4210,
    "output_tokens": 890,
    "cache_read_input_tokens": 3000,
    "cache_creation_input_tokens": 0,
}


def fake(outputs):
    client = FakeClient(outputs)
    P.set_client(client)
    return client


@pytest.fixture(autouse=True)
def reset_client():
    yield
    P.set_client(None)


HISTORY = [
    {"id": "n001", "family": "model", "hypothesis": "FM baseline",
     "status": "ok", "accepted": True, "metrics": {"primary": 0.6016}},
    {"id": "n002", "family": "feature", "hypothesis": "add user_ctr",
     "status": "error", "errors": [{"error_class": "schema"}]},
]
PARENT = HISTORY[0]


# --- the acceptance criteria ------------------------------------------------

def test_returns_a_valid_action_from_a_fake_history():
    fake([(VALID_ACTION, USAGE)])
    action, _ = P.propose(HISTORY, "knowledge", PARENT)
    assert isinstance(action, Action)
    assert action.config.loss == "lambdarank"


def test_token_usage_is_captured():
    """Total tokens is a scored deliverable — it must come back on every call."""
    fake([(VALID_ACTION, USAGE)])
    _, usage = P.propose(HISTORY, "knowledge", PARENT)
    assert usage["in"] == 4210
    assert usage["out"] == 890
    assert usage["model"] == "claude-opus-5"
    assert usage["cache_read"] == 3000


def test_two_tier_routing():
    """Opus proposes, Haiku repairs (Section 12)."""
    client = fake([(VALID_ACTION, USAGE)])
    P.propose(HISTORY, "knowledge", PARENT)
    assert client.messages.calls[0]["model"] == P.PROPOSE_MODEL == "claude-opus-5"

    client = fake([(VALID_ACTION, USAGE)])
    P.repair(VALID_ACTION, {"error_class": "syntax", "traceback": "boom"})
    assert client.messages.calls[0]["model"] == P.REPAIR_MODEL == "claude-haiku-4-5"


def test_usage_records_the_model_that_actually_ran():
    fake([(VALID_ACTION, USAGE)])
    _, usage = P.repair(VALID_ACTION, {"error_class": "syntax"})
    assert usage["model"] == "claude-haiku-4-5"


# --- prompt construction ----------------------------------------------------

def test_knowledge_is_in_the_cached_system_prefix():
    """Caching is a prefix match: stable content first, volatile after."""
    client = fake([(VALID_ACTION, USAGE)])
    P.propose(HISTORY, "KNOWLEDGE-MARKER", PARENT)
    system = client.messages.calls[0]["system"]
    assert "KNOWLEDGE-MARKER" in system[0]["text"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_volatile_context_is_in_the_user_turn_not_the_system_prefix():
    client = fake([(VALID_ACTION, USAGE)])
    P.propose(HISTORY, "knowledge", PARENT)
    call = client.messages.calls[0]
    assert "n002" not in call["system"][0]["text"]
    assert "n002" in call["messages"][0]["content"]


def test_history_is_summarised_not_dumped_whole():
    """A full node carries config, segments, gates and a traceback. Sending 40
    of those back costs thousands of tokens per iteration for nothing."""
    node = {
        "id": "n003", "family": "model", "hypothesis": "h", "status": "ok",
        "metrics": {"primary": 0.61, "gauc": 0.68},
        "segments": {"activity_q1": 0.58}, "gates": {"contract": True},
        "config": {"model": "fm", "hparams": {"k": 16}},
        "reasoning": "x" * 5000,
    }
    [row] = P.summarise_history([node])
    assert row["primary"] == 0.61
    assert "segments" not in row and "gates" not in row and "reasoning" not in row


def test_error_class_survives_summarisation():
    """The proposer needs to know *why* a node died to avoid repeating it."""
    rows = P.summarise_history(HISTORY)
    assert rows[1]["error_class"] == "schema"


def test_families_covered_is_reported_to_the_model():
    """A7 enforces diversity; telling the proposer helps it comply."""
    client = fake([(VALID_ACTION, USAGE)])
    P.propose(HISTORY, "knowledge", PARENT)
    context = json.loads(client.messages.calls[0]["messages"][0]["content"].split("\n\n", 1)[1])
    assert context["families_covered"] == {
        "feature": 1, "model": 1, "objective": 0, "training": 0, "ensemble": 0
    }


def test_adaptive_thinking_on_propose():
    client = fake([(VALID_ACTION, USAGE)])
    P.propose(HISTORY, "knowledge", PARENT)
    assert client.messages.calls[0]["thinking"] == {"type": "adaptive"}


# --- consistency checks the schema cannot express ---------------------------

@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(type="tune", family="training", search_space=None), "search_space"),
        (dict(type="code", family="feature", code=None), "code"),
        (dict(type="config", family="model", config=None), "config"),
    ],
)
def test_inconsistent_actions_are_rejected(kwargs, message):
    action = Action(hypothesis="h", reasoning="r", parent="n001", **kwargs)
    with pytest.raises(P.ProposeError, match=message):
        P.check_action_consistency(action)


def test_blend_needs_two_parents():
    action = Action(
        hypothesis="h", reasoning="r", type="blend", family="ensemble", parent="n001",
        config={"model": "blend", "parents": ["n001"]},
    )
    with pytest.raises(P.ProposeError, match="two parents"):
        P.check_action_consistency(action)


def test_valid_action_passes_consistency():
    P.check_action_consistency(VALID_ACTION)  # must not raise


# --- retry behaviour --------------------------------------------------------

def test_inconsistent_proposal_is_re_asked_once_then_succeeds():
    bad = Action(hypothesis=VALID_ACTION.hypothesis, reasoning="r", type="tune", family="training",
                 parent="n001", search_space=None)
    client = fake([(bad, USAGE), (VALID_ACTION, USAGE)])
    action, usage = P.propose(HISTORY, "knowledge", PARENT)
    assert action is VALID_ACTION
    assert len(client.messages.calls) == 2
    assert "search_space" in client.messages.calls[1]["messages"][0]["content"]
    assert bad.hypothesis in client.messages.calls[1]["messages"][0]["content"]


def test_retry_usage_is_summed_not_overwritten():
    """Both attempts are billed, so both must reach the resource report."""
    bad = Action(hypothesis=VALID_ACTION.hypothesis, reasoning="r", type="code", family="feature",
                 parent="n001", code=None)
    fake([(bad, USAGE), (VALID_ACTION, USAGE)])
    _, usage = P.propose(HISTORY, "knowledge", PARENT)
    assert usage["in"] == 2 * 4210
    assert usage["out"] == 2 * 890


def test_gives_up_after_max_attempts():
    bad = Action(hypothesis="h", reasoning="r", type="code", family="feature",
                 parent="n001", code=None)
    fake([(bad, USAGE)] * P.MAX_ATTEMPTS)
    with pytest.raises(P.ProposeError, match="no valid Action") as exc:
        P.propose(HISTORY, "knowledge", PARENT)
    assert exc.value.usage["in"] == P.MAX_ATTEMPTS * USAGE["input_tokens"]
    assert exc.value.usage["out"] == P.MAX_ATTEMPTS * USAGE["output_tokens"]


def test_schema_retry_cannot_silently_change_the_hypothesis():
    bad = Action(hypothesis="original claim", reasoning="r", type="code",
                 family="feature", parent="n001", code=None)
    changed = VALID_ACTION.model_copy(update={"hypothesis": "different claim"})
    fake([(bad, USAGE), (changed, USAGE)])
    with pytest.raises(P.ProposeError, match="changed the hypothesis"):
        P.propose(HISTORY, "knowledge", PARENT)


def test_api_failure_becomes_a_propose_error_not_a_raw_exception():
    fake([(RuntimeError("connection reset"), USAGE)])
    with pytest.raises(P.ProposeError, match="connection reset"):
        P.propose(HISTORY, "knowledge", PARENT)


# --- repair invariants ------------------------------------------------------

@pytest.mark.parametrize("field", ["hypothesis", "family", "parent"])
def test_repair_cannot_change_experiment_identity(field):
    replacements = {
        "hypothesis": "a different claim",
        "family": "training",
        "parent": "n999",
    }
    changed = VALID_ACTION.model_copy(update={field: replacements[field]})
    fake([(changed, USAGE)])
    with pytest.raises(P.ProposeError, match=f"preserve {field}") as exc:
        P.repair(VALID_ACTION, {"error_class": "syntax"})
    assert exc.value.usage["in"] == USAGE["input_tokens"]


def test_repair_may_change_mechanism_without_changing_claim():
    repaired = VALID_ACTION.model_copy(
        update={"config": VALID_ACTION.config.model_copy(update={"loss": "pointwise"})}
    )
    fake([(repaired, USAGE)])
    result, _ = P.repair(VALID_ACTION, {"error_class": "schema"})
    assert result.config.loss == "pointwise"
    assert result.hypothesis == VALID_ACTION.hypothesis


# --- prompt files -----------------------------------------------------------

def test_prompts_load_and_carry_the_corrected_label():
    """The plan's prose says `click`; the shipped evaluator says `long_view`.
    A prompt with the wrong label would steer every proposal wrong."""
    propose_md = P.load_prompt("propose")
    assert "long_view" in propose_md
    assert "0.8645" in propose_md, "oracle ceiling must be stated"
    assert "0.0008" in propose_md, "noise floor must be stated"


def test_repair_prompt_forbids_changing_the_hypothesis():
    assert "hypothesis" in P.load_prompt("repair")


def test_knowledge_file_loads():
    assert "long_view" in P.load_knowledge()
