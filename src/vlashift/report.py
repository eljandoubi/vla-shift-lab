"""Markdown tables and figures for the study, from a tree of finished runs."""

import json
from pathlib import Path

from vlashift.config import ALL_ARMS
from vlashift.results import Run, leftovers, load_tree
from vlashift.stats import (
    mcnemar_exact,
    paired_counts,
    per_task_differences,
    task_bootstrap,
    wilson,
)
from vlashift.suites import SUITES

BASELINE = "keep"
LABELS = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long",
}
SCOPES = (*SUITES, "all")
IMAGE_ARMS = ("noise_0.1", "occl_0.4", "bright_0.5")
START = "<!-- results:start -->"
END = "<!-- results:end -->"


def ordered_arms(tree: dict[str, dict[str, Run]]) -> list[str]:
    return [a for a in ALL_ARMS if a in tree] + sorted(a for a in tree if a not in ALL_ARMS)


def episodes_in(runs: dict[str, Run], scope: str) -> dict[tuple, bool]:
    """Outcomes keyed (suite, task, episode) for one suite or, with scope 'all', every suite."""
    chosen = SUITES if scope == "all" else (scope,)
    return {
        (suite, *key): flag
        for suite in chosen
        if suite in runs
        for key, flag in runs[suite].outcomes.items()
    }


def cell(successes: int, trials: int) -> str:
    if trials == 0:
        return "not run"
    low, high = wilson(successes, trials)
    rate = 100 * successes / trials
    return f"{rate:.1f} [{100 * low:.1f}, {100 * high:.1f}] ({successes}/{trials})"


def results_table(tree: dict[str, dict[str, Run]]) -> str:
    columns = [LABELS[s] for s in SUITES] + ["All"]
    rows = ["| arm | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for arm in ordered_arms(tree):
        cells = []
        for scope in SCOPES:
            flags = episodes_in(tree[arm], scope)
            cells.append(cell(sum(flags.values()), len(flags)))
        rows.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def paired_rows(tree: dict[str, dict[str, Run]], n_boot: int = 10_000) -> list[dict]:
    """Each arm against keep on the episodes both ran, per suite and over all suites."""
    if BASELINE not in tree:
        return []
    rows = []
    for arm in ordered_arms(tree):
        if arm == BASELINE:
            continue
        for scope in SCOPES:
            base, other = episodes_in(tree[BASELINE], scope), episodes_in(tree[arm], scope)
            counts = paired_counts(base, other)
            if counts.pairs == 0:
                continue
            mean, low, high = task_bootstrap(per_task_differences(base, other), n_boot=n_boot)
            rows.append(
                {
                    "arm": arm,
                    "scope": scope,
                    "pairs": counts.pairs,
                    "delta": 100 * mean,
                    "low": 100 * low,
                    "high": 100 * high,
                    "keep_only": counts.only_a,
                    "arm_only": counts.only_b,
                    "p": mcnemar_exact(counts.only_a, counts.only_b),
                }
            )
    return rows


def format_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def paired_table(rows: list[dict]) -> str:
    if not rows:
        return "No paired comparison yet: it needs the keep arm and one more arm on a suite."
    lines = [
        "| arm | scope | pairs | keep minus arm (points) | 95% CI, tasks resampled "
        "| keep only / arm only | p, exact McNemar, uncorrected |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['arm']} | {LABELS.get(r['scope'], 'All')} | {r['pairs']} | {r['delta']:+.1f} "
            f"| [{r['low']:+.1f}, {r['high']:+.1f}] | {r['keep_only']} / {r['arm_only']} "
            f"| {format_p(r['p'])} |"
        )
    return "\n".join(lines)


def delta_points(tree: dict[str, dict[str, Run]], arm: str, scope: str) -> float | None:
    """Success of keep minus success of arm in points, on the episodes both ran."""
    if BASELINE not in tree or arm not in tree:
        return None
    counts = paired_counts(episodes_in(tree[BASELINE], scope), episodes_in(tree[arm], scope))
    if counts.pairs == 0:
        return None
    return 100 * (counts.only_a - counts.only_b) / counts.pairs


def h1_verdict(spatial: float, obj: float, goal: float) -> str:
    """H1 with the thresholds of the README: what the blank instruction costs on each suite."""
    worst = max(spatial, obj)
    if worst >= 30 or goal <= worst:
        return "falsified"
    if worst < 15 and goal - worst >= 15:
        return "holds"
    return "unclear"


def h2_verdict(spatial: float, obj: float) -> str:
    """H2: what the instruction of another task costs on Spatial and Object."""
    worst = max(spatial, obj)
    if worst >= 20:
        return "falsified"
    return "holds" if worst < 10 else "unclear"


Interval = tuple[float, float, float]


def h3_verdict(noise: Interval, occl: Interval, bright: Interval) -> str:
    """H3 from (difference, low, high) over all suites for noise, occlusion and brightness."""
    if bright[0] >= min(noise[0], occl[0]):
        return "falsified"
    if min(noise[1], occl[1]) > bright[2]:
        return "holds"
    return "unclear"


def signed(value: float) -> str:
    return f"{value:+.1f}"


def suite_deltas(tree: dict[str, dict[str, Run]], arm: str, suites: tuple[str, ...]) -> dict:
    """Suite label -> keep minus arm in points, for the suites where both ran."""
    found = {LABELS[s]: delta_points(tree, arm, s) for s in suites}
    return {label: value for label, value in found.items() if value is not None}


def listed(arm: str, deltas: dict[str, float]) -> str:
    if not deltas:
        return ""
    return f"{arm}: " + ", ".join(f"{label} {signed(value)}" for label, value in deltas.items())


def hypothesis_rows(tree: dict[str, dict[str, Run]], rows: list[dict] | None = None) -> list[tuple]:
    """(claim, numbers, verdict) for H1 to H3; a hypothesis whose arms did not run says so."""
    rows = paired_rows(tree) if rows is None else rows
    blank = suite_deltas(tree, "blank", ("libero_spatial", "libero_object", "libero_goal"))
    other = suite_deltas(tree, "other", ("libero_spatial", "libero_object"))
    overall = {r["arm"]: r for r in rows if r["scope"] == "all"}
    image = {arm: overall[arm] for arm in IMAGE_ARMS if arm in overall}
    image_numbers = ", ".join(
        f"{arm} {signed(r['delta'])} [{signed(r['low'])}, {signed(r['high'])}]"
        for arm, r in image.items()
    )
    # H3 compares arms over the same episodes: a partial run of one arm would bias it.
    same_episodes = len(image) == 3 and len({r["pairs"] for r in image.values()}) == 1
    return [
        (
            "H1: the instruction matters little where the scene fixes the task",
            listed("blank", blank),
            h1_verdict(*blank.values()) if len(blank) == 3 else "not run",
        ),
        (
            "H2: a wrong instruction changes nothing where the policy follows its visual habit",
            listed("other", other),
            h2_verdict(*other.values()) if len(other) == 2 else "not run",
        ),
        (
            "H3: noise and occlusion hurt more than brightness",
            image_numbers,
            h3_verdict(*[(r["delta"], r["low"], r["high"]) for r in image.values()])
            if same_episodes
            else "not run",
        ),
    ]


def hypothesis_table(rows: list[tuple]) -> str:
    lines = ["| hypothesis | points, keep minus arm | verdict |", "|---|---|---|"]
    lines += [f"| {claim} | {numbers} | {verdict} |" for claim, numbers, verdict in rows]
    return "\n".join(lines)


def timing_summary(tree: dict[str, dict[str, Run]]) -> str:
    runs = [r for arms in tree.values() for r in arms.values() if r.seconds_per_episode]
    if not runs:
        return "No timing in the runs."
    episodes = sum(r.episodes for r in runs)
    seconds = sum(r.seconds_per_episode * r.episodes for r in runs)
    return (
        f"{len(runs)} runs, {episodes} episodes, {seconds / episodes:.1f} s per episode on "
        f"average, {seconds / 3600:.1f} h in total."
    )


def figures(tree: dict[str, dict[str, Run]], directory: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, suite in zip(axes.flat, SUITES, strict=True):
        names, rates, below, above = [], [], [], []
        for arm in ordered_arms(tree):
            flags = episodes_in(tree[arm], suite)
            if not flags:
                continue
            k, n = sum(flags.values()), len(flags)
            low, high = wilson(k, n)
            names.append(arm)
            rates.append(100 * k / n)
            below.append(100 * (k / n - low))
            above.append(100 * (high - k / n))
        axis.barh(range(len(names)), rates, xerr=[below, above], color="#4c72b0", capsize=2)
        axis.set_yticks(range(len(names)), names)
        axis.invert_yaxis()
        axis.set_xlim(0, 100)
        axis.set_title(LABELS[suite])
    fig.supxlabel("success rate, % (95% Wilson interval)")
    fig.tight_layout()
    fig.savefig(directory / "success_by_arm.png", dpi=150)
    plt.close(fig)

    rows = [r for r in paired_rows(tree, n_boot=2000) if r["scope"] == "all"]
    if rows:
        fig, axis = plt.subplots(figsize=(7, 0.45 * len(rows) + 1.5))
        deltas = [r["delta"] for r in rows]
        errors = [[r["delta"] - r["low"] for r in rows], [r["high"] - r["delta"] for r in rows]]
        axis.errorbar(deltas, range(len(rows)), xerr=errors, fmt="o", color="#c44e52", capsize=3)
        axis.axvline(0, color="grey", linewidth=0.8)
        axis.set_yticks(range(len(rows)), [r["arm"] for r in rows])
        axis.invert_yaxis()
        axis.set_xlabel("keep minus arm, points (95% CI, tasks resampled)")
        fig.tight_layout()
        fig.savefig(directory / "drop_vs_keep.png", dpi=150)
        plt.close(fig)


def write_all(runs: Path, out: Path) -> str:
    """Write tables.md, summary.json and the figures under out; return the results table."""
    tree = load_tree(runs)
    if not tree:
        raise ValueError(f"no finished run under {runs}")
    out.mkdir(parents=True, exist_ok=True)
    table = results_table(tree)
    rows = paired_rows(tree)
    verdicts = hypothesis_table(hypothesis_rows(tree, rows))
    parts = [
        "## Success rate in %, [95% Wilson interval] (successes/episodes)",
        table,
        "## Hypotheses, with the thresholds written in the README",
        verdicts,
        "## Paired comparison with keep",
        paired_table(rows),
        "## Timing",
        timing_summary(tree),
    ]
    unfinished = leftovers(runs)
    if unfinished:
        parts += ["## Unfinished runs, not counted", "\n".join(f"- {name}" for name in unfinished)]
    (out / "tables.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    cells = {
        arm: {s: {"successes": r.successes, "episodes": r.episodes} for s, r in suites.items()}
        for arm, suites in tree.items()
    }
    (out / "summary.json").write_text(
        json.dumps({"cells": cells, "paired": rows}, indent=2) + "\n", encoding="utf-8"
    )
    figures(tree, out / "figures")
    return f"{table}\n\n{verdicts}"


def update_readme(readme: Path, table: str) -> None:
    text = readme.read_text(encoding="utf-8")
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError(f"{readme} needs exactly one {START} and one {END} marker")
    head, rest = text.split(START)
    _, tail = rest.split(END)
    readme.write_text(f"{head}{START}\n{table}\n{END}{tail}", encoding="utf-8")
