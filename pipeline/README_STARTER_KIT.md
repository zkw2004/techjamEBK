# `evaluate.py` and `submit.py` are missing on purpose

Both files are **copied verbatim from the organiser starter kit** and are the
scoring ground truth (Section 1.2, rule 3; task D1).

Do not write them by hand, do not reimplement them from memory, do not edit
them after copying. Their SHA-256 hashes go into `logs/manifest.json` at
preflight, and preflight **fails closed** on mismatch (Section 8.3).

To install them:

```bash
cp <starter-kit>/evaluate.py pipeline/evaluate.py
cp <starter-kit>/submit.py   pipeline/submit.py
shasum -a 256 pipeline/evaluate.py pipeline/submit.py
```

Then record those hashes as the Day-0 baseline (Section 10.1) and build the
typed wrapper in `agent/manifest.py` (D2) against them.

The raw KuaiRand-Pure data goes in `data/`, which is git-ignored.
