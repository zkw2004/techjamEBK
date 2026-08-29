"""The five Day-1 probes. Task D5.

Single command runs all five and prints a comparison table. THE OUTPUT
DETERMINES THE SECTION 6.7 MODEL LADDER ORDERING — do not assume an ordering
in advance (Section 6.4; Dacrema 2019, Rendle 2020).

Total compute budget: under one hour.
"""

from __future__ import annotations

PROBES = {
    "P1": "FM + 5 aggregate features — does feature engineering alone beat baseline?",
    "P2": "LightGBM pointwise, same features — does GBDT beat FM?",
    "P3": "LightGBM lambdarank, same features — does a ranking loss actually help?",
    "P4": "DeepFM, light tuning — is the neural branch worth pursuing?",
    "P5": "FM + Optuna, 30 trials — was the baseline simply undertuned?",
}


def main() -> None:
    raise NotImplementedError("D5")


if __name__ == "__main__":
    main()
