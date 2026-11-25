"""Hooks that the patched lerobot-eval script calls, and the record of what they did."""

import atexit
import importlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from vlashift import perturb
from vlashift.config import RuntimeConfig
from vlashift.instructions import transform

IMAGE_PREFIX = "observation.images."
STATE_KEY = "observation.state"
FIRST_STATES = 3
LIBERO_MODULE = "lerobot.sim.integrations.libero"


def _is_tensor(value) -> bool:
    return hasattr(value, "detach")


def _to_array(value) -> np.ndarray:
    return value.detach().cpu().numpy() if _is_tensor(value) else np.asarray(value)


class Runtime:
    def __init__(self, config: RuntimeConfig):
        if config.dump_frames and not config.applied_path:
            raise ValueError(
                "VLASHIFT_DUMP_FRAMES needs VLASHIFT_APPLIED_PATH to say where to write"
            )
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.counts = {"instructions_changed": 0, "images_perturbed": 0, "init_states_perturbed": 0}
        self.first_states: list[list[float]] = []
        self.frames_saved = 0

    def instructions(self, texts):
        before = list(texts)
        after = transform(self.config.instruction, before)
        self.counts["instructions_changed"] += sum(
            a != b for a, b in zip(before, after, strict=True)
        )
        return after

    def images(self, observation):
        """Shift the camera images of a batch as the policy is about to see them."""
        self._remember_state(observation)
        keys = [k for k in observation if k.startswith(IMAGE_PREFIX) and self._wanted(k)]
        out = dict(observation)
        if self.config.image_shift:
            if not keys:
                raise KeyError(f"an image shift is set but {sorted(observation)} has no image")
            for key in keys:
                out[key] = self._shift(observation[key])
            self.counts["images_perturbed"] += 1
        self._dump(out, keys)
        return out

    def init_states(self, states, key):
        out = perturb.noisy_joints(states, self.config.joint_std, self.config.seed, key)
        self.counts["init_states_perturbed"] += len(out)
        return out

    def _wanted(self, key):
        cameras = self.config.cameras
        return cameras is None or key[len(IMAGE_PREFIX) :] in cameras

    def _shift(self, value):
        array = _to_array(value)
        if not np.issubdtype(array.dtype, np.floating) or (array.size and array.max() > 1.001):
            raise ValueError("image shifts expect float images in [0, 1]")
        config = self.config
        if config.brightness is not None:
            array = perturb.scale_brightness(array, config.brightness)
        if config.noise_std is not None:
            array = perturb.add_noise(array, config.noise_std, self.rng)
        if config.occlusion is not None:
            array = perturb.occlude(array, config.occlusion)
        return value.new_tensor(array) if _is_tensor(value) else array

    def _remember_state(self, observation):
        if STATE_KEY in observation and len(self.first_states) < FIRST_STATES:
            state = np.atleast_2d(_to_array(observation[STATE_KEY]))[0].astype(float)
            self.first_states.append(np.round(state, 4).tolist())

    def _dump(self, observation, keys):
        if not keys or self.frames_saved >= self.config.dump_frames:
            return
        from PIL import Image

        folder = Path(self.config.applied_path).parent / "frames"
        folder.mkdir(parents=True, exist_ok=True)
        for key in keys:
            chw = np.clip(_to_array(observation[key])[0], 0.0, 1.0)
            pixels = (chw * 255).round().astype(np.uint8).transpose(1, 2, 0)
            name = f"frame{self.frames_saved}_{key[len(IMAGE_PREFIX) :]}.png"
            Image.fromarray(pixels).save(folder / name)
        self.frames_saved += 1

    def record(self) -> dict:
        return {
            "config": asdict(self.config),
            "counts": dict(self.counts),
            "first_states": self.first_states,
            "frames_saved": self.frames_saved,
        }

    def write_applied(self) -> None:
        if self.config.applied_path is None:
            return
        path = Path(self.config.applied_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.record(), indent=2), encoding="utf-8")


_runtime: Runtime | None = None


def get() -> Runtime:
    global _runtime
    if _runtime is None:
        _runtime = Runtime(RuntimeConfig.from_environ())
    return _runtime


def reset(environ=None) -> Runtime:
    global _runtime
    _runtime = Runtime(RuntimeConfig.from_environ(environ))
    return _runtime


def instructions(texts):
    return get().instructions(texts)


def images(observation):
    return get().images(observation)


def video_episodes() -> int:
    return get().config.video_episodes


def install(module_name: str = LIBERO_MODULE, at_exit: bool = True) -> None:
    """Wrap the LIBERO init-state loader when joint noise is set, and save the record at exit."""
    runtime = get()
    if runtime.config.joint_std is not None:
        module = importlib.import_module(module_name)
        original = module.get_task_init_states

        def with_joint_noise(task_suite, index, *args, **kwargs):
            states = original(task_suite, index, *args, **kwargs)
            return runtime.init_states(states, task_suite.tasks[index].name)

        module.get_task_init_states = with_joint_noise
    if at_exit:
        atexit.register(runtime.write_applied)
