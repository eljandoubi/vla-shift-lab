"""Pixel and robot-state perturbations as pure numpy functions.

Images are float arrays in [0, 1] with height and width on the last two axes (channel first, as
the policy receives them). A LIBERO init state is a flattened simulator state: time, the 7 arm
joint angles, the 2 finger joints, then the objects.
"""

import hashlib

import numpy as np

ARM_JOINTS = slice(1, 8)
JOINT_COUNT = 7


def scale_brightness(images: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError(f"brightness scale must be positive, got {scale}")
    return np.clip(images * scale, 0.0, 1.0)


def add_noise(images: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std < 0:
        raise ValueError(f"noise std must not be negative, got {std}")
    noisy = np.clip(images + rng.normal(0.0, std, size=images.shape), 0.0, 1.0)
    return noisy.astype(images.dtype, copy=False)


def occlude(images: np.ndarray, side_fraction: float, fill: float = 0.0) -> np.ndarray:
    """Cover a centred square whose side is a fraction of the shorter image side."""
    if not 0.0 <= side_fraction <= 1.0:
        raise ValueError(f"patch side fraction must be in [0, 1], got {side_fraction}")
    height, width = images.shape[-2:]
    side = round(side_fraction * min(height, width))
    top, left = (height - side) // 2, (width - side) // 2
    out = images.copy()
    out[..., top : top + side, left : left + side] = fill
    return out


def noisy_joints(states: np.ndarray, std: float, seed: int, key: str) -> np.ndarray:
    """Add Gaussian noise (radians) to the arm joints of every init state, nothing else.

    The draw of a row depends on (seed, key, row) only, so the same state gets the same
    direction at every level and a level is an exact rescaling of the others.
    """
    if std < 0:
        raise ValueError(f"joint noise std must not be negative, got {std}")
    if states.ndim != 2 or states.shape[1] < 10:
        raise ValueError(f"expected init states of shape (n, >=10), got {states.shape}")
    key_int = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    out = np.array(states, copy=True)
    for row in range(len(out)):
        draw = np.random.default_rng([seed, key_int, row]).standard_normal(JOINT_COUNT)
        out[row, ARM_JOINTS] += std * draw
    return out
