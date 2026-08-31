"""Ablation-driven drawer targeting. Task A10.

The loop's weakness is not that it lacks ideas, it is that it has no evidence
about *where* the current best solution is actually sensitive. Rotating through
families on a schedule spends iterations uniformly on components that may
contribute nothing. MLE-STAR's finding is that cheap ablations on the incumbent
localise the headroom first, and exploration concentrates there.

At this scale an ablation round is nearly free: a screen-tier run is seconds to
tens of seconds, so probing the incumbent's components costs about as much as
one wasted proposal — and unlike a wasted proposal it produces a table the
proposer can reason from, which is also exactly what "what the agent chose to
target and why" is graded on.

Ablation probes are measurements, not experiments: they are never persisted to
the node ledger and never eligible for promotion. Only their summary rides on
the node that was probed.
"""

from __future__ import annotations

from typing import Any

from agent.schema import FAMILIES, SUPPORTED_LOSSES, Action

# One base run plus this many variants must fit the A10 budget of ~5 minutes at
# screen fidelity. A screen run is roughly 25s on the real split, so 8 leaves
# headroom; the official five fields plus a loss swap is 6 variants.
MAX_VARIANTS = 8

# Boolean hparams that switch an architectural block on, each independently
# ablatable by switching it back off. Both are identity-at-initialisation
# (pipeline/models/deepfm.py), so the off-variant is the same network minus that
# one mechanism rather than a differently-initialised one.
ARCHITECTURE_BLOCKS = ("lhuc", "senet")

ABLATION_HYPOTHESIS = (
    "Ablation probe of the incumbent: measure how much each component "
    "contributes, so the next proposal targets the sensitive one."
)

ABLATION_REASONING = (
    "Not a candidate. Disabling one component at a time and re-screening "
    "localises where the incumbent's score actually comes from; the resulting "
    "sensitivity table is injected into the next proposal."
)


def _probe_action(node: dict, config: dict) -> Action:
    """Wrap one ablated config as an Action the ordinary executor can run."""
    family = node.get("family")
    return Action(
        hypothesis=ABLATION_HYPOTHESIS,
        reasoning=ABLATION_REASONING,
        type="config",
        family=family if family in FAMILIES else "model",
        parent=node.get("id") or "n000",
        config=config,
    )


def ablation_variants(
    config: dict, parent_configs: dict[str, dict] | None = None
) -> list[tuple[str, dict]]:
    """``(component, ablated_config)`` pairs, each isolating one component.

    Every variant must still be a runnable config — an ablation that cannot
    execute measures nothing. So a single-feature config yields no feature
    variants (dropping the last field leaves no model), and a loss swap is only
    offered for a loss the model genuinely implements.
    """
    variants: list[tuple[str, dict]] = []
    model = config.get("model")
    features = list(config.get("features") or [])

    if len(features) > 1:
        for name in features:
            reduced = [item for item in features if item != name]
            variants.append((f"feature:{name}", {**config, "features": reduced}))

    supported = sorted(SUPPORTED_LOSSES.get(str(model), frozenset()))
    current = config.get("loss", "pointwise")
    alternative = next((option for option in supported if option != current), None)
    if alternative is not None:
        variants.append((f"loss:{alternative}", {**config, "loss": alternative}))

    # Architectural blocks that are off by default and switched on by an hparam
    # (C11 LHUC, C12 SENet). Only an *enabled* block is worth a variant: turning
    # one off measures what it contributes, whereas turning an absent one on
    # would be proposing a new experiment, which is not what an ablation is for.
    hparams = config.get("hparams") or {}
    for block in ARCHITECTURE_BLOCKS:
        if hparams.get(block):
            variants.append(
                (f"block:{block}", {**config, "hparams": {**hparams, block: False}})
            )

    # For a blend, "dropping a member" cannot stay a blend — the runner requires
    # exactly two parents. The measurable question is what each parent scores
    # alone, which is the same quantity a member-drop would report.
    if model == "blend" and parent_configs:
        for parent_id, parent_config in parent_configs.items():
            if parent_config.get("model"):
                variants.append((f"ensemble:{parent_id}", dict(parent_config)))

    return variants


def _blend_parent_configs(config: dict) -> dict[str, dict]:
    """Resolve a blend's parent configs from the ledger, tolerating misses."""
    from agent import store

    resolved: dict[str, dict] = {}
    for parent_id in config.get("parents") or []:
        try:
            parent = store.read(str(parent_id))
        except (OSError, ValueError, KeyError):
            continue
        parent_config = parent.get("config") or {}
        if parent_config.get("model"):
            resolved[str(parent_id)] = parent_config
    return resolved


def _primary(result: dict) -> float | None:
    if result.get("status") != "ok":
        return None
    value = (result.get("metrics") or {}).get("primary")
    return float(value) if isinstance(value, (int, float)) else None


def ablate(
    node: dict,
    *,
    execute_fn: Any,
    max_variants: int = MAX_VARIANTS,
) -> dict | None:
    """Screen the incumbent and one variant per component; rank by sensitivity.

    ``execute_fn(action, fidelity) -> node dict`` is injected so the loop can
    pass its own executor and tests can pass a fixture. Returns ``None`` when
    there is nothing to measure — no config, an unscoreable base run, or no
    runnable variants — because a partial table is worse than no table: the
    proposer would read absent components as insensitive ones.

    Sensitivity is ``abs(base - ablated)``: a component whose removal moves the
    score in either direction is carrying signal. Sign is kept separately in
    ``delta`` so a component that *hurts* is visible as such.
    """
    config = node.get("config") or {}
    if not config.get("model"):
        return None

    parent_configs = _blend_parent_configs(config) if config.get("model") == "blend" else None
    variants = ablation_variants(config, parent_configs)[:max_variants]
    if not variants:
        return None

    base_primary = _primary(execute_fn(_probe_action(node, config), "screen"))
    if base_primary is None:
        return None

    components = []
    for component, variant_config in variants:
        ablated = _primary(execute_fn(_probe_action(node, variant_config), "screen"))
        if ablated is None:
            continue  # a variant that will not run tells us nothing; skip it
        components.append(
            {
                "component": component,
                "primary": ablated,
                "delta": ablated - base_primary,
                "sensitivity": abs(ablated - base_primary),
            }
        )
    if not components:
        return None

    components.sort(key=lambda item: item["sensitivity"], reverse=True)
    # No node id here: the ledger is append-only, so this table is written as a
    # field *of* the node it describes rather than referring to it by id.
    return {"base_primary": base_primary, "components": components}


def render_sensitivity_table(ablation: dict | None) -> str:
    """Markdown for the propose prompt. Empty string when there is no table."""
    if not ablation or not ablation.get("components"):
        return ""
    lines = [
        "## Component sensitivity of the current best node",
        "",
        f"Screen-tier base primary {ablation['base_primary']:.6f}. "
        "Sensitivity is how far the score moves when that component is removed — "
        "a high value means the incumbent depends on it, a value near zero means "
        "it is inert and changing it is unlikely to be worth an iteration.",
        "",
        "Read this as evidence about where to look, not as a verdict. These are "
        "**screen-tier** runs on capped budgets (reduced epochs and trees), so a "
        "component that needs a full budget to pay off — a sparse high-cardinality "
        "embedding especially — can look inert here while mattering at full "
        "fidelity. A surprising zero is a hypothesis worth testing, not a "
        "settled fact.",
        "",
        "| component | screen primary | delta vs base | sensitivity |",
        "|---|---|---|---|",
    ]
    for item in ablation["components"]:
        lines.append(
            f"| {item['component']} | {item['primary']:.6f} | "
            f"{item['delta']:+.6f} | {item['sensitivity']:.6f} |"
        )
    return "\n".join(lines)


def unused_features(ablation: dict | None, config: dict) -> list[str]:
    """Registered features the incumbent does not use, most promising first.

    Feeds A11's hedge queue. The ablation table only names features already in
    the config, so "top unused" is drawn from the registry: anything the
    incumbent has not tried yet is a candidate, and the ones already proven
    inert elsewhere are not re-suggested.
    """
    from pipeline.features import FEATURES

    inert = {
        item["component"].removeprefix("feature:")
        for item in (ablation or {}).get("components", [])
        if item["component"].startswith("feature:") and item["sensitivity"] == 0.0
    }
    used = set(config.get("features") or [])
    return [name for name in sorted(FEATURES) if name not in used and name not in inert]
