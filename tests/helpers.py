"""Fabricated lerobot-eval outputs in the layout of LeRobot 0.7.0 (eval_policy_all)."""

import json
from pathlib import Path


def make_info(suite, flags_by_task, eval_ep_s=7.5):
    per_task = []
    for task_id, flags in flags_by_task.items():
        k = sum(flags)
        per_task.append(
            {
                "task_group": suite,
                "task_id": task_id,
                "metrics": {
                    "sum_rewards": [float(f) for f in flags],
                    "max_rewards": [float(f) for f in flags],
                    "successes": list(flags),
                    "video_paths": [],
                    "predicted_video_paths": [],
                },
                "n_episodes": len(flags),
                "n_success": k,
                "pc_success": 100.0 * k / len(flags),
                "pc_success_ci95": [0.0, 100.0],
            }
        )
    every = [f for flags in flags_by_task.values() for f in flags]
    group = {
        "avg_sum_reward": sum(every) / len(every),
        "avg_max_reward": sum(every) / len(every),
        "pc_success": 100.0 * sum(every) / len(every),
        "n_episodes": len(every),
        "n_success": sum(every),
        "pc_success_ci95": [0.0, 100.0],
        "video_paths": [],
        "predicted_video_paths": [],
    }
    return {
        "per_task": per_task,
        "per_group": {suite: group},
        "overall": {**group, "eval_s": eval_ep_s * len(every), "eval_ep_s": eval_ep_s},
    }


def write_run(root, arm, suite, flags_by_task, eval_ep_s=7.5):
    directory = Path(root) / arm / suite
    directory.mkdir(parents=True)
    (directory / "eval_info.json").write_text(
        json.dumps(make_info(suite, flags_by_task, eval_ep_s), indent=2)
    )
    return directory


def flags(successes, episodes=10):
    """A task outcome list with the given number of successes first."""
    return [True] * successes + [False] * (episodes - successes)


def uniform(successes_per_task, tasks=10, episodes=10):
    return {t: flags(successes_per_task, episodes) for t in range(tasks)}
