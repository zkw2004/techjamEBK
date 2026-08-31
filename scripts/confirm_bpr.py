"""One-off confirm-tier check of n004's BPR config (5 seeds), run manually
outside the agent loop to resolve whether its +0.0013 delta over baseline
is signal or noise, without spending any propose()/repair() API tokens."""

import json

from pipeline.train import run_experiment

CONFIG = {
    "model": "fm",
    "loss": "pairwise",
    "features": ["user_id", "video_id", "author_id", "tab", "dur_bucket"],
    "negative_sampling": "all",
    "hparams": {},
}


def main() -> None:
    result = run_experiment(CONFIG, fidelity="confirm", seed=42, timeout_s=1800)
    print("status:", result.get("status"))
    print("primary:", result.get("primary"))
    print("gauc:", result.get("gauc"))
    print("ndcg:", result.get("ndcg"))
    print("fold_primaries:", result.get("fold_primaries"))
    print("seconds:", result.get("seconds"))
    if result.get("status") != "ok":
        print("traceback:", result.get("traceback"))
    bulky = ("val_scores", "val_user_ids", "test_scores")
    dump = {k: v for k, v in result.items() if k not in bulky}
    with open("/tmp/bpr_confirm_result.json", "w") as f:
        json.dump(dump, f, indent=2, default=str)


if __name__ == "__main__":
    main()
