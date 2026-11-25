import json
import subprocess
import sys
from pathlib import Path

import pytest
from helpers import flags, uniform, write_run

from vlashift.report import (
    cell,
    episodes_in,
    figures,
    format_p,
    ordered_arms,
    paired_rows,
    paired_table,
    results_table,
    timing_summary,
    update_readme,
    write_all,
)
from vlashift.results import load_tree

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tree(tmp_path):
    for suite in ("libero_spatial", "libero_goal"):
        write_run(tmp_path, "keep", suite, uniform(9), eval_ep_s=8.0)
        write_run(tmp_path, "blank", suite, uniform(1), eval_ep_s=6.0)
    write_run(tmp_path, "noise_0.1", "libero_goal", {t: flags(9) for t in range(10)})
    write_run(tmp_path, "mystery", "libero_goal", uniform(5))
    return load_tree(tmp_path)


def test_ordered_arms_follow_the_canonical_order_then_unknown_names(tree):
    assert ordered_arms(tree) == ["keep", "blank", "noise_0.1", "mystery"]


def test_episodes_in_a_suite_or_everywhere(tree):
    one = episodes_in(tree["keep"], "libero_goal")
    assert len(one) == 100
    assert ("libero_goal", 3, 9) in one
    assert len(episodes_in(tree["keep"], "all")) == 200
    assert episodes_in(tree["noise_0.1"], "libero_spatial") == {}


def test_cell_text():
    assert cell(90, 100) == "90.0 [82.6, 94.5] (90/100)"
    assert cell(0, 0) == "not run"


def test_results_table_has_one_row_per_arm_and_marks_missing_cells(tree):
    lines = results_table(tree).splitlines()
    assert lines[0] == "| arm | Spatial | Object | Goal | Long | All |"
    assert [line.split("|")[1].strip() for line in lines[2:]] == [
        "keep",
        "blank",
        "noise_0.1",
        "mystery",
    ]
    keep = lines[2].split("|")
    assert keep[1].strip() == "keep"
    assert keep[2].strip() == "90.0 [82.6, 94.5] (90/100)"
    assert keep[3].strip() == "not run"
    assert keep[5].strip() == "not run"
    assert keep[6].strip() == "90.0 [85.1, 93.4] (180/200)"


def test_paired_rows_count_discordant_episodes_and_take_the_difference(tree):
    rows = {(r["arm"], r["scope"]): r for r in paired_rows(tree, n_boot=500)}
    blank = rows[("blank", "libero_spatial")]
    assert blank["pairs"] == 100
    assert (blank["keep_only"], blank["arm_only"]) == (80, 0)
    assert blank["delta"] == pytest.approx(80.0)
    assert (blank["low"], blank["high"]) == (pytest.approx(80.0), pytest.approx(80.0))
    assert blank["p"] < 1e-20
    assert rows[("blank", "all")]["pairs"] == 200
    noise = rows[("noise_0.1", "libero_goal")]
    assert (noise["keep_only"], noise["arm_only"], noise["delta"]) == (0, 0, 0.0)
    assert noise["p"] == 1.0


def test_paired_rows_skip_scopes_without_common_episodes(tree):
    scopes = {r["scope"] for r in paired_rows(tree, n_boot=100) if r["arm"] == "noise_0.1"}
    assert scopes == {"libero_goal", "all"}


def test_paired_rows_need_the_baseline(tmp_path):
    write_run(tmp_path, "blank", "libero_goal", uniform(1))
    assert paired_rows(load_tree(tmp_path)) == []
    assert "No paired comparison" in paired_table([])


def test_paired_table_formats_each_row(tree):
    text = paired_table(paired_rows(tree, n_boot=200))
    header, rule, first = text.splitlines()[:3]
    assert "keep minus arm (points)" in header
    assert "uncorrected" in header
    assert rule.count("---") == 7
    assert first.startswith("| blank | Spatial | 100 | +80.0 | [+80.0, +80.0] | 80 / 0 | <0.001 |")


@pytest.mark.parametrize(("p", "text"), [(0.0004, "<0.001"), (0.0123, "0.012"), (1.0, "1.000")])
def test_format_p(p, text):
    assert format_p(p) == text


def test_timing_summary_weights_runs_by_episodes(tree):
    text = timing_summary(tree)
    assert text.startswith("6 runs, 600 episodes")
    assert "7.2 s per episode" in text
    assert "1.2 h in total" in text


def test_timing_summary_without_timing(tmp_path):
    directory = write_run(tmp_path, "keep", "libero_goal", uniform(9))
    info = json.loads((directory / "eval_info.json").read_text())
    del info["overall"]["eval_ep_s"]
    (directory / "eval_info.json").write_text(json.dumps(info))
    assert timing_summary(load_tree(tmp_path)) == "No timing in the runs."


def test_write_all_creates_tables_summary_and_figures(tmp_path):
    runs = tmp_path / "runs"
    for suite in ("libero_spatial", "libero_goal"):
        write_run(runs, "keep", suite, uniform(9))
        write_run(runs, "blank", suite, uniform(1))
    (runs / "blank" / "libero_object.partial").mkdir()
    out = tmp_path / "out"
    table = write_all(runs, out)
    text = (out / "tables.md").read_text()
    assert table in text
    assert "## Paired comparison with keep" in text
    assert "blank/libero_object.partial" in text
    summary = json.loads((out / "summary.json").read_text())
    assert summary["cells"]["keep"]["libero_goal"] == {"successes": 90, "episodes": 100}
    assert {r["arm"] for r in summary["paired"]} == {"blank"}
    assert (out / "figures" / "success_by_arm.png").stat().st_size > 1000
    assert (out / "figures" / "drop_vs_keep.png").stat().st_size > 1000


def test_figures_without_a_baseline_only_draw_the_first_chart(tmp_path):
    write_run(tmp_path / "runs", "blank", "libero_goal", uniform(1))
    figures(load_tree(tmp_path / "runs"), tmp_path / "fig")
    assert (tmp_path / "fig" / "success_by_arm.png").exists()
    assert not (tmp_path / "fig" / "drop_vs_keep.png").exists()


def test_write_all_refuses_an_empty_tree(tmp_path):
    with pytest.raises(ValueError, match="no finished run"):
        write_all(tmp_path, tmp_path / "out")


def test_update_readme_replaces_only_the_marked_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- results:start -->\nold\n<!-- results:end -->\noutro\n")
    update_readme(readme, "| new |")
    assert (
        readme.read_text()
        == "intro\n<!-- results:start -->\n| new |\n<!-- results:end -->\noutro\n"
    )


@pytest.mark.parametrize(
    "text", ["no markers", "<!-- results:start -->\n<!-- results:start -->\n<!-- results:end -->"]
)
def test_update_readme_needs_exactly_one_pair_of_markers(tmp_path, text):
    readme = tmp_path / "README.md"
    readme.write_text(text)
    with pytest.raises(ValueError, match="marker"):
        update_readme(readme, "x")


def test_make_tables_script_end_to_end(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "keep", "libero_goal", uniform(9))
    write_run(runs, "other", "libero_goal", uniform(2))
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n")
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_tables.py"), "--runs", str(runs),
         "--out", str(tmp_path / "res"), "--readme", str(readme)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert "| other |" in readme.read_text()
    assert (tmp_path / "res" / "tables.md").exists()
