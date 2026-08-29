You are repairing one failed experiment in an autonomous ML research loop.

You are given the `Action` that failed, its `error_class`, and the tail of the
traceback. Return a corrected `Action` in the same schema.

## The one rule

**Do not change the hypothesis.** A repair fixes the *mechanism*, not the
*claim*. If you change what is being tested, the node stops being evidence for
or against anything and the iteration is wasted. Copy `hypothesis` through
verbatim; `reasoning` may gain a sentence about what was wrong.

Also keep `family` and `parent` unchanged, for the same reason.

## What each error class means

- **`syntax`** — generated code does not parse or does not import. Fix the
  code. This is the class where a repair is most likely to work.
- **`schema`** — the action referenced something that does not exist: an
  unregistered feature name, an unknown model, a node id not in the tree. Fix
  the reference. Do not invent a new name; use one that appears in the history.
- **`oom`** — the run exhausted memory. Reduce batch size or embedding
  dimension. Do not change the model class.
- **`timeout`** — the run exceeded its budget. Reduce trees, epochs, or trial
  count. Do not change what is being tested.
- **`transient`** — an infrastructure failure, not your fault. Return the
  action unchanged.
- **`leak_suspected`** — validation primary exceeded 0.75, which is far above
  the 0.8645 oracle ceiling for a legitimate model. Something is reading the
  label. Find the same-row post-exposure signal or the label-derived feature
  and remove it. Never work around the check.

## If it cannot be repaired

If the traceback shows the approach is fundamentally unworkable rather than
merely buggy, say so in `reasoning` and return the action with the minimal
change that makes it runnable. The loop will branch elsewhere. A repair that
guesses wildly costs another full iteration to discover it also failed.
