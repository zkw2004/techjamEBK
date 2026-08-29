# Repair prompt (A5)

TODO: fill in. Routed to Haiku 4.5, one attempt.

- Inputs: the failed `Action`, the `error_class`, and the traceback.
- Output: a corrected `Action` in the same schema, or an explicit give-up.
- Do not change the hypothesis while repairing — a repair fixes the
  mechanism, not the claim. Changing the claim makes the node uninterpretable.
