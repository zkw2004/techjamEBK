# Propose prompt (A3)

TODO: fill in. Structure to hit:

- Role: you propose ONE experiment against a frozen typed contract.
- Inputs interpolated: `{knowledge}`, `{node_history}`, `{parent_node}`,
  `{families_covered}`, `{budget_remaining}`.
- Output: a single JSON object matching `agent.schema.Action`. Nothing else.
- `hypothesis` is graded and copied verbatim into the run log. It must be a
  falsifiable claim about this dataset, not a restatement of the config.
- Cover all five families (feature, model, objective, training, ensemble)
  before refining any one of them twice.
- Worked examples: Appendix B of AGENT_PLAN.md.
