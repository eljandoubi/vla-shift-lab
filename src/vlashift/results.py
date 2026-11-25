"""Read the output directories of lerobot-eval into paired episode outcomes.

Layout: <root>/<arm>/<suite>/eval_info.json, one directory per arm and suite. Inside the file,
`per_task` holds one entry per task with the success flag of every episode, in episode order, so
episode j of a task has the same seed and init state in every arm.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from vlashift.suites import SUITES

Outcomes = dict[tuple[int, int], bool]


@dataclass(frozen=True)
class Run:
    arm: str
    suite: str
    outcomes: Outcomes
    seconds_per_episode: float | None
    path: Path

    @property
    def episodes(self) -> int:
        return len(self.outcomes)

    @property
    def successes(self) -> int:
        return sum(self.outcomes.values())


def read_run(directory: Path) -> Run:
    path = directory / "eval_info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    entries = info.get("per_task") or []
    if not entries:
        raise ValueError(f"{path}: no per_task entries")
    suite = directory.name
    outcomes: Outcomes = {}
    for entry in entries:
        try:
            group = entry["task_group"]
            task_id = int(entry["task_id"])
            flags = entry["metrics"]["successes"]
        except KeyError as missing:
            raise ValueError(f"{path}: a per_task entry has no {missing}") from None
        if group != suite:
            raise ValueError(f"{path}: results of {group} stored in the {suite} directory")
        for episode, flag in enumerate(flags):
            if (task_id, episode) in outcomes:
                raise ValueError(f"{path}: task {task_id} appears twice")
            outcomes[(task_id, episode)] = bool(flag)
    seconds = (info.get("overall") or {}).get("eval_ep_s")
    return Run(directory.parent.name, suite, outcomes, seconds, directory)


def load_tree(root: Path) -> dict[str, dict[str, Run]]:
    """Every finished run under root, as {arm: {suite: Run}}; other directories are skipped."""
    tree: dict[str, dict[str, Run]] = {}
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for suite in SUITES:
            if (arm_dir / suite / "eval_info.json").is_file():
                tree.setdefault(arm_dir.name, {})[suite] = read_run(arm_dir / suite)
    return tree


def leftovers(root: Path) -> list[str]:
    """Runs that started and did not finish: <arm>/<suite>.partial or .failed directories."""
    found = []
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        found += [
            f"{arm_dir.name}/{p.name}"
            for p in sorted(arm_dir.iterdir())
            if p.is_dir() and p.suffix in {".partial", ".failed"}
        ]
    return found
