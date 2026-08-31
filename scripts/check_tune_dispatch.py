"""Manual end-to-end check of execute()'s type="tune" dispatch.

Run as a file, not via `python3 -c`/stdin: pipeline.train spawns worker
processes on macOS, and spawn re-imports __main__ by path.
"""

from agent.execute import execute
from agent.schema import Action

BASE_CONFIG = {
    "model": "fm",
    "features": ["user_id", "video_id", "author_id", "tab", "dur_bucket"],
    "hparams": {"k": 8, "lr": 0.005, "max_epochs": 1},
}


def _action(budget: int = 2) -> Action:
    return Action(
        hypothesis="verify tune dispatch reaches Optuna and returns a tuned config",
        reasoning="manual verification of the A-side tune hook",
        type="tune",
        family="training",
        parent="n001",
        config=dict(BASE_CONFIG),
        search_space={"lr": ["loguniform", 1e-4, 1e-2]},
        budget=budget,
    )


def main() -> None:
    smoke = execute(_action(), fidelity="smoke", timeout_s=600)
    print("smoke status :", smoke["status"])
    print("smoke tuning :", smoke.get("tuning"), "(expected None: smoke never searches)")
    if smoke["status"] != "ok":
        print(smoke["errors"][0]["traceback"][-800:])
        return

    screen = execute(_action(), fidelity="screen", timeout_s=1800)
    print("screen status:", screen["status"])
    if screen["status"] != "ok":
        print(screen["errors"][0]["traceback"][-1500:])
        return
    print("screen tuning:", screen.get("tuning"))
    print("recorded lr  :", screen["config"]["hparams"].get("lr"))
    print("base lr      :", BASE_CONFIG["hparams"]["lr"])
    print("primary      :", screen["metrics"].get("primary"))


if __name__ == "__main__":
    main()
