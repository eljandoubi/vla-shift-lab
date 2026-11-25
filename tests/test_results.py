import json

import pytest
from helpers import flags, make_info, uniform, write_run

from vlashift.results import leftovers, load_tree, read_run


def test_read_run_collects_episode_outcomes_per_task(tmp_path):
    directory = write_run(tmp_path, "keep", "libero_spatial", {0: flags(9), 1: flags(3)})
    run = read_run(directory)
    assert (run.arm, run.suite) == ("keep", "libero_spatial")
    assert run.episodes == 20
    assert run.successes == 12
    assert run.outcomes[(0, 8)] is True
    assert run.outcomes[(0, 9)] is False
    assert run.outcomes[(1, 2)] is True
    assert run.outcomes[(1, 3)] is False
    assert run.seconds_per_episode == 7.5


def test_read_run_keeps_task_ids_of_a_subset(tmp_path):
    directory = write_run(tmp_path, "keep", "libero_goal", {4: flags(10), 7: flags(0)})
    run = read_run(directory)
    assert sorted({task for task, _ in run.outcomes}) == [4, 7]


def test_read_run_ignores_the_order_of_per_task_entries(tmp_path):
    directory = write_run(tmp_path, "keep", "libero_goal", {2: flags(1), 0: flags(5)})
    info = json.loads((directory / "eval_info.json").read_text())
    info["per_task"].reverse()
    (directory / "eval_info.json").write_text(json.dumps(info))
    run = read_run(directory)
    assert run.outcomes[(0, 4)] is True
    assert run.outcomes[(2, 0)] is True
    assert run.outcomes[(2, 1)] is False


def test_read_run_rejects_a_file_without_tasks(tmp_path):
    directory = tmp_path / "keep" / "libero_goal"
    directory.mkdir(parents=True)
    (directory / "eval_info.json").write_text(json.dumps({"per_task": [], "overall": {}}))
    with pytest.raises(ValueError, match="no per_task"):
        read_run(directory)


def test_read_run_rejects_an_entry_without_successes(tmp_path):
    directory = tmp_path / "keep" / "libero_goal"
    directory.mkdir(parents=True)
    info = make_info("libero_goal", {0: flags(3)})
    del info["per_task"][0]["metrics"]["successes"]
    (directory / "eval_info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="successes"):
        read_run(directory)


def test_read_run_rejects_results_of_another_suite(tmp_path):
    directory = tmp_path / "keep" / "libero_goal"
    directory.mkdir(parents=True)
    (directory / "eval_info.json").write_text(json.dumps(make_info("libero_object", {0: flags(3)})))
    with pytest.raises(ValueError, match="libero_object stored in the libero_goal"):
        read_run(directory)


def test_read_run_rejects_a_task_listed_twice(tmp_path):
    directory = tmp_path / "keep" / "libero_goal"
    directory.mkdir(parents=True)
    info = make_info("libero_goal", {0: flags(3)})
    info["per_task"].append(info["per_task"][0])
    (directory / "eval_info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="twice"):
        read_run(directory)


def test_read_run_without_timing_gives_none(tmp_path):
    directory = write_run(tmp_path, "keep", "libero_goal", {0: flags(3)})
    info = json.loads((directory / "eval_info.json").read_text())
    del info["overall"]["eval_ep_s"]
    (directory / "eval_info.json").write_text(json.dumps(info))
    assert read_run(directory).seconds_per_episode is None


def test_load_tree_groups_runs_by_arm_and_suite(tmp_path):
    write_run(tmp_path, "keep", "libero_spatial", uniform(9))
    write_run(tmp_path, "keep", "libero_goal", uniform(8))
    write_run(tmp_path, "blank", "libero_goal", uniform(1))
    tree = load_tree(tmp_path)
    assert sorted(tree) == ["blank", "keep"]
    assert sorted(tree["keep"]) == ["libero_goal", "libero_spatial"]
    assert tree["blank"]["libero_goal"].successes == 10


def test_load_tree_skips_unfinished_directories_and_loose_files(tmp_path):
    write_run(tmp_path, "keep", "libero_spatial", uniform(9))
    (tmp_path / "keep" / "libero_goal.partial").mkdir()
    (tmp_path / "keep" / "libero_goal.partial" / "eval_info.json").write_text("{}")
    (tmp_path / "noise_0.1" / "libero_object").mkdir(parents=True)
    (tmp_path / "notes.txt").write_text("x")
    tree = load_tree(tmp_path)
    assert list(tree) == ["keep"]
    assert list(tree["keep"]) == ["libero_spatial"]


def test_leftovers_lists_partial_and_failed_runs(tmp_path):
    write_run(tmp_path, "keep", "libero_spatial", uniform(9))
    (tmp_path / "keep" / "libero_goal.partial").mkdir()
    (tmp_path / "blank" / "libero_object.failed").mkdir(parents=True)
    assert leftovers(tmp_path) == ["blank/libero_object.failed", "keep/libero_goal.partial"]
