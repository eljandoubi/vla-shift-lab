"""Feed every instruction arm through a random-weight SmolVLA on CPU (needs LeRobot and torch).

Random weights say nothing about success. The check is that each arm's text goes through the
real tokenizer and select_action without a shape error, that the arms really give the model
different tokens, and that the image hook keeps torch tensors on their dtype and device.

    PYTHONPATH=src python scripts/plumbing_check.py results/plumbing_check.json
"""

import json
import sys
import time
from pathlib import Path

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from vlashift import runtime
from vlashift.config import parse_arm
from vlashift.instructions import MODES
from vlashift.suites import instructions

IMAGE, STATE, ACTION, CAMERAS = 256, 8, 7, 2
TEXT = instructions()["libero_goal"][3]


def build_policy():
    config = SmolVLAConfig(device="cpu", load_vlm_weights=False)
    features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE,))}
    for i in range(CAMERAS):
        shape = (3, IMAGE, IMAGE)
        features[f"observation.images.camera{i + 1}"] = PolicyFeature(FeatureType.VISUAL, shape)
    config.input_features = features
    config.output_features = {"action": PolicyFeature(FeatureType.ACTION, (ACTION,))}
    return config, SmolVLAPolicy(config).eval()


def run_arm(policy, config, tokenizer, arm):
    text = runtime.reset(parse_arm(arm).env()).instructions([TEXT])[0]
    encoded = tokenizer(
        [text + "\n"],
        padding="max_length",
        max_length=config.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    generator = torch.Generator().manual_seed(0)
    batch = {
        key: torch.rand(1, 3, IMAGE, IMAGE, generator=generator) for key in config.image_features
    }
    batch["observation.state"] = torch.randn(1, STATE, generator=generator)
    batch["observation.language.tokens"] = encoded["input_ids"]
    batch["observation.language.attention_mask"] = encoded["attention_mask"].bool()
    torch.manual_seed(0)
    policy.reset()
    start = time.time()
    with torch.no_grad():
        action = policy.select_action(batch)
    return {
        "arm": arm,
        "text": text,
        "tokens": int(encoded["attention_mask"].sum()),
        "action_shape": list(action.shape),
        "action_head": [round(float(x), 4) for x in action[0, :3]],
        "seconds": round(time.time() - start, 1),
    }


def check_image_hook():
    rt = runtime.reset(parse_arm("noise_0.1").env())
    value = torch.full((2, 3, 64, 64), 0.5, dtype=torch.float32)
    out = rt.images({"observation.images.image": value})["observation.images.image"]
    return {
        "is_tensor": isinstance(out, torch.Tensor),
        "dtype_kept": out.dtype == value.dtype,
        "device_kept": out.device == value.device,
        "changed": bool((out != value).any()),
        "in_range": bool(out.min() >= 0 and out.max() <= 1),
    }


def main(out_path):
    torch.set_num_threads(4)
    started = time.time()
    config, policy = build_policy()
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    rows = [run_arm(policy, config, tokenizer, arm) for arm in MODES]
    report = {
        "torch": torch.__version__,
        "build_and_run_seconds": round(time.time() - started, 1),
        "arms": rows,
        "all_actions_have_shape_1x7": all(r["action_shape"] == [1, ACTION] for r in rows),
        "distinct_action_heads": len({tuple(r["action_head"]) for r in rows}),
        "token_counts": {r["arm"]: r["tokens"] for r in rows},
        "image_hook": check_image_hook(),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/plumbing_check.json")
