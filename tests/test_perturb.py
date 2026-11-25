import numpy as np
import pytest

from vlashift.perturb import add_noise, noisy_joints, occlude, scale_brightness


@pytest.fixture
def images():
    return np.random.default_rng(0).random((2, 3, 32, 32), dtype=np.float32)


def test_brightness_scales_and_clips(images):
    dark = scale_brightness(images, 0.5)
    assert np.allclose(dark, images * 0.5)
    bright = scale_brightness(images, 2.0)
    assert bright.max() == 1.0
    assert np.allclose(bright, np.clip(images * 2.0, 0.0, 1.0))


def test_brightness_one_is_identity_and_keeps_dtype(images):
    out = scale_brightness(images, 1.0)
    assert np.array_equal(out, images)
    assert out.dtype == np.float32


def test_brightness_does_not_touch_the_input(images):
    before = images.copy()
    scale_brightness(images, 0.3)
    assert np.array_equal(images, before)


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_brightness_rejects_non_positive_scale(images, scale):
    with pytest.raises(ValueError, match="positive"):
        scale_brightness(images, scale)


def test_noise_has_the_requested_spread(images):
    flat = np.full((1, 3, 128, 128), 0.5, dtype=np.float32)
    noisy = add_noise(flat, 0.05, np.random.default_rng(1))
    assert abs(float((noisy - flat).std()) - 0.05) < 0.003
    assert abs(float((noisy - flat).mean())) < 0.002


def test_noise_stays_in_range_and_keeps_dtype(images):
    noisy = add_noise(images, 0.5, np.random.default_rng(2))
    assert noisy.min() >= 0.0
    assert noisy.max() <= 1.0
    assert noisy.dtype == np.float32
    assert noisy.shape == images.shape


def test_noise_zero_std_is_identity(images):
    assert np.array_equal(add_noise(images, 0.0, np.random.default_rng(3)), images)


def test_noise_is_reproducible_from_the_generator(images):
    first = add_noise(images, 0.1, np.random.default_rng(4))
    second = add_noise(images, 0.1, np.random.default_rng(4))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, add_noise(images, 0.1, np.random.default_rng(5)))


def test_noise_rejects_negative_std(images):
    with pytest.raises(ValueError, match="negative"):
        add_noise(images, -0.1, np.random.default_rng(0))


def test_occlusion_covers_a_centred_square():
    ones = np.ones((1, 3, 100, 100), dtype=np.float32)
    out = occlude(ones, 0.4)
    assert out[0, 0, 30:70, 30:70].max() == 0.0
    assert out[0, 0, :30].min() == 1.0
    assert out[0, 0, 70:].min() == 1.0
    assert out[0, 0, :, :30].min() == 1.0
    assert out[0, 0, :, 70:].min() == 1.0
    assert out.sum() == ones.sum() - 3 * 40 * 40


def test_occlusion_side_uses_the_shorter_edge():
    ones = np.ones((3, 60, 100), dtype=np.float32)
    out = occlude(ones, 0.5)
    assert int((out == 0).sum()) == 3 * 30 * 30


def test_occlusion_extremes_and_fill_value():
    ones = np.ones((3, 20, 20), dtype=np.float32)
    assert np.array_equal(occlude(ones, 0.0), ones)
    assert occlude(ones, 1.0).max() == 0.0
    assert np.all(occlude(ones, 0.5, fill=0.25)[:, 10, 10] == 0.25)


def test_occlusion_leaves_the_input_alone_and_rejects_bad_sizes():
    ones = np.ones((3, 20, 20), dtype=np.float32)
    occlude(ones, 0.5)
    assert ones.min() == 1.0
    with pytest.raises(ValueError, match="fraction"):
        occlude(ones, 1.5)
    with pytest.raises(ValueError, match="fraction"):
        occlude(ones, -0.1)


@pytest.fixture
def states():
    base = np.zeros((50, 47))
    base[:, 1:8] = [0.0, -0.2, 0.0, -2.4, 0.0, 2.2, 0.8]
    base[:, 8:10] = [0.02, -0.02]
    base[:, 10:] = np.linspace(-1, 1, 37)
    return base


def test_joint_noise_changes_only_the_seven_arm_columns(states):
    out = noisy_joints(states, 0.1, seed=0, key="task")
    assert np.array_equal(out[:, 0], states[:, 0])
    assert np.array_equal(out[:, 8:], states[:, 8:])
    assert not np.any(out[:, 1:8] == states[:, 1:8])


def test_joint_noise_zero_std_is_identity(states):
    assert np.array_equal(noisy_joints(states, 0.0, seed=0, key="task"), states)


def test_joint_noise_has_the_requested_spread():
    big = np.zeros((4000, 12))
    out = noisy_joints(big, 0.1, seed=3, key="k")
    assert abs(float(out[:, 1:8].std()) - 0.1) < 0.004
    assert abs(float(out[:, 1:8].mean())) < 0.004


def test_joint_noise_levels_are_rescalings_of_one_draw(states):
    small = noisy_joints(states, 0.05, seed=1, key="task") - states
    large = noisy_joints(states, 0.2, seed=1, key="task") - states
    assert np.allclose(large, 4 * small)


def test_joint_noise_depends_on_seed_and_key_and_row(states):
    a = noisy_joints(states, 0.1, seed=1, key="task")
    assert np.array_equal(a, noisy_joints(states, 0.1, seed=1, key="task"))
    assert not np.array_equal(a, noisy_joints(states, 0.1, seed=2, key="task"))
    assert not np.array_equal(a, noisy_joints(states, 0.1, seed=1, key="other task"))
    assert not np.array_equal(a[0], a[1])


def test_joint_noise_row_does_not_depend_on_the_other_rows(states):
    full = noisy_joints(states, 0.1, seed=1, key="task")
    head = noisy_joints(states[:5], 0.1, seed=1, key="task")
    assert np.array_equal(full[:5], head)


def test_joint_noise_does_not_modify_its_input(states):
    before = states.copy()
    noisy_joints(states, 0.3, seed=0, key="task")
    assert np.array_equal(states, before)


def test_joint_noise_rejects_other_layouts(states):
    with pytest.raises(ValueError, match="init states"):
        noisy_joints(states[0], 0.1, seed=0, key="task")
    with pytest.raises(ValueError, match="init states"):
        noisy_joints(np.zeros((5, 6)), 0.1, seed=0, key="task")
    with pytest.raises(ValueError, match="negative"):
        noisy_joints(states, -0.1, seed=0, key="task")
