import sys
import types

import numpy as np
import pytest
from PIL import Image

from vlashift import runtime
from vlashift.config import parse_arm
from vlashift.instructions import other
from vlashift.suites import instructions as table

TEXT = table()["libero_goal"][7]


class FakeTensor:
    """The few tensor methods the hooks use, backed by numpy."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def new_tensor(self, data):
        return FakeTensor(np.array(data, dtype=self.array.dtype))


def images_batch(value=0.5, batch=2):
    return np.full((batch, 3, 32, 32), value, dtype=np.float32)


def observation(**extra):
    base = {
        "observation.images.image": images_batch(),
        "observation.images.image2": images_batch(0.25),
        "observation.state": np.arange(16, dtype=np.float32).reshape(2, 8),
        "task": ["pick"],
    }
    return {**base, **extra}


def make(arm="keep", **extra):
    environ = {**parse_arm(arm).env(), **{k: str(v) for k, v in extra.items()}}
    return runtime.Runtime(runtime.RuntimeConfig.from_environ(environ))


def test_keep_leaves_instructions_alone_and_counts_nothing():
    rt = make("keep")
    assert rt.instructions([TEXT]) == [TEXT]
    assert rt.counts["instructions_changed"] == 0


@pytest.mark.parametrize("arm", ["blank", "other", "shuffled", "paraphrase"])
def test_instruction_arms_change_every_text(arm):
    rt = make(arm)
    texts = [TEXT, table()["libero_spatial"][0]]
    out = rt.instructions(texts)
    assert all(a != b for a, b in zip(texts, out, strict=True))
    assert rt.counts["instructions_changed"] == 2


def test_other_arm_gives_the_next_task_instruction():
    assert make("other").instructions([TEXT]) == [other(TEXT)]


def test_unknown_text_stops_the_other_arm():
    with pytest.raises(KeyError):
        make("other").instructions(["wave"])


def test_keep_arm_returns_the_images_untouched():
    rt = make("keep")
    obs = observation()
    out = rt.images(obs)
    assert out["observation.images.image"] is obs["observation.images.image"]
    assert rt.counts["images_perturbed"] == 0


def test_brightness_scales_every_camera_and_does_not_edit_the_input():
    rt = make("bright_0.5")
    obs = observation()
    out = rt.images(obs)
    assert np.allclose(out["observation.images.image"], 0.25)
    assert np.allclose(out["observation.images.image2"], 0.125)
    assert np.allclose(obs["observation.images.image"], 0.5)
    assert out["observation.state"] is obs["observation.state"]
    assert out["task"] == ["pick"]
    assert rt.counts["images_perturbed"] == 1


def test_noise_is_reproducible_for_a_seed_and_differs_between_seeds():
    first = make("noise_0.1", VLASHIFT_SEED=3).images(observation())
    second = make("noise_0.1", VLASHIFT_SEED=3).images(observation())
    third = make("noise_0.1", VLASHIFT_SEED=4).images(observation())
    key = "observation.images.image"
    assert np.array_equal(first[key], second[key])
    assert not np.array_equal(first[key], third[key])
    assert not np.array_equal(first[key], observation()[key])


def test_occlusion_blacks_out_the_centre_only():
    out = make("occl_0.5").images(observation())["observation.images.image"]
    assert out[0, 0, 16, 16] == 0.0
    assert out[0, 0, 0, 0] == 0.5


def test_camera_option_restricts_the_shift():
    out = make("bright_0.5", VLASHIFT_CAMERAS="image2").images(observation())
    assert np.allclose(out["observation.images.image"], 0.5)
    assert np.allclose(out["observation.images.image2"], 0.125)


def test_a_shift_without_images_fails_loudly():
    with pytest.raises(KeyError, match="no image"):
        make("noise_0.1").images({"observation.state": np.zeros((1, 8))})
    with pytest.raises(KeyError, match="no image"):
        make("noise_0.1", VLASHIFT_CAMERAS="wrist").images(observation())


@pytest.mark.parametrize(
    "bad", [np.zeros((1, 3, 8, 8), dtype=np.uint8), np.full((1, 3, 8, 8), 255.0, dtype=np.float32)]
)
def test_images_outside_the_unit_range_are_refused(bad):
    with pytest.raises(ValueError, match="float images in"):
        make("bright_0.5").images({"observation.images.image": bad})


def test_tensor_like_values_keep_their_type():
    obs = {
        "observation.images.image": FakeTensor(images_batch()),
        "observation.state": FakeTensor(np.zeros((1, 8))),
    }
    out = make("bright_0.5").images(obs)["observation.images.image"]
    assert isinstance(out, FakeTensor)
    assert out.array.dtype == np.float32
    assert np.allclose(out.array, 0.25)


def test_first_states_are_recorded_up_to_three_calls():
    rt = make("keep")
    for _ in range(5):
        rt.images(observation())
    assert len(rt.first_states) == 3
    assert rt.first_states[0] == list(map(float, range(8)))


def test_frames_are_written_for_the_first_calls_only(tmp_path):
    rt = make(
        "noise_0.1", VLASHIFT_DUMP_FRAMES=1, VLASHIFT_APPLIED_PATH=tmp_path / "a" / "applied.json"
    )
    rt.images(observation())
    rt.images(observation())
    files = sorted(p.name for p in (tmp_path / "a" / "frames").iterdir())
    assert files == ["frame0_image.png", "frame0_image2.png"]
    assert Image.open(tmp_path / "a" / "frames" / "frame0_image.png").size == (32, 32)
    assert rt.frames_saved == 1


def test_frames_of_the_keep_arm_are_unperturbed(tmp_path):
    rt = make("keep", VLASHIFT_DUMP_FRAMES=1, VLASHIFT_APPLIED_PATH=tmp_path / "applied.json")
    rt.images(observation())
    pixels = np.asarray(Image.open(tmp_path / "frames" / "frame0_image.png"))
    assert pixels.max() == 128


def test_dumping_frames_needs_a_place_to_write():
    with pytest.raises(ValueError, match="APPLIED_PATH"):
        make("keep", VLASHIFT_DUMP_FRAMES=1)


def test_write_applied_saves_config_and_counts(tmp_path):
    import json

    path = tmp_path / "deep" / "applied.json"
    rt = make("blank", VLASHIFT_APPLIED_PATH=path)
    rt.instructions([TEXT])
    rt.write_applied()
    saved = json.loads(path.read_text())
    assert saved["config"]["instruction"] == "blank"
    assert saved["counts"] == {
        "instructions_changed": 1,
        "images_perturbed": 0,
        "init_states_perturbed": 0,
    }


def test_write_applied_without_a_path_does_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make("blank").write_applied()
    assert list(tmp_path.iterdir()) == []


def fake_libero(monkeypatch, name="fake_libero_env"):
    states = np.zeros((6, 47))
    states[:, 1:8] = 0.1
    calls = []

    def get_task_init_states(task_suite, index, is_libero_plus=False):
        calls.append((index, is_libero_plus))
        return states.copy()

    module = types.ModuleType(name)
    module.get_task_init_states = get_task_init_states
    monkeypatch.setitem(sys.modules, name, module)
    return module, states, calls


def suite(names):
    return types.SimpleNamespace(tasks=[types.SimpleNamespace(name=n) for n in names])


def test_install_wraps_the_init_state_loader_when_joint_noise_is_set(monkeypatch):
    module, states, calls = fake_libero(monkeypatch)
    rt = runtime.reset(parse_arm("joint_0.2").env())
    runtime.install("fake_libero_env", at_exit=False)
    out = module.get_task_init_states(suite(["a", "b"]), 1, is_libero_plus=True)
    assert calls == [(1, True)]
    assert not np.any(out[:, 1:8] == states[:, 1:8])
    assert np.array_equal(out[:, 8:], states[:, 8:])
    assert rt.counts["init_states_perturbed"] == 6


def test_the_noise_of_a_task_depends_on_its_name(monkeypatch):
    module, _, _ = fake_libero(monkeypatch)
    runtime.reset(parse_arm("joint_0.2").env())
    runtime.install("fake_libero_env", at_exit=False)
    first = module.get_task_init_states(suite(["a", "b"]), 0)
    second = module.get_task_init_states(suite(["a", "b"]), 1)
    again = module.get_task_init_states(suite(["a", "b"]), 0)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, second)


def test_install_leaves_the_loader_alone_without_joint_noise(monkeypatch):
    module, _, _ = fake_libero(monkeypatch)
    original = module.get_task_init_states
    runtime.reset(parse_arm("noise_0.1").env())
    runtime.install("fake_libero_env", at_exit=False)
    assert module.get_task_init_states is original


def test_install_registers_the_record_for_exit(monkeypatch):
    registered = []
    monkeypatch.setattr(runtime.atexit, "register", registered.append)
    rt = runtime.reset(parse_arm("keep").env())
    runtime.install()
    assert registered == [rt.write_applied]


def test_module_level_hooks_use_the_environment(monkeypatch):
    monkeypatch.setenv("VLASHIFT_INSTRUCTION", "blank")
    monkeypatch.setenv("VLASHIFT_VIDEOS", "2")
    runtime.reset()
    assert runtime.instructions([TEXT]) == [""]
    assert runtime.video_episodes() == 2
    assert runtime.images(observation())["observation.images.image"].shape == (2, 3, 32, 32)
    assert runtime.get() is runtime.get()
