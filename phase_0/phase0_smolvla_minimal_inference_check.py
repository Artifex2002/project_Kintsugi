"""
Phase 0.1 - SmolVLA Minimal Inference Script
============================================

This script:
1. Loads SmolVLA-450M from HuggingFace
2. Takes a text instruction + dummy image
3. Gets action vector output
4. Confirms output shape for 7-DOF robot arm

Reference Docs: 
https://huggingface.co/lerobot/smolvla_base
https://huggingface.co/docs/lerobot/introduction_processors
https://huggingface.co/docs/lerobot/env_processor

"""
import json
import torch
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# ── Config ---────────────────────────────────────────────────────────────────

MODEL_ID   = "lerobot/smolvla_base"
DATASET_ID = "lerobot/libero"
OUTPUT_DIR = Path("check_script_outputs")

# smolvla_base was trained with these 3 camera keys.
# LIBERO only has 2 cameras, so we map image→camera1, image2→camera2,
# and duplicate image2 into camera3 as a placeholder.
CAMERA_KEY_MAP = {
    "observation.images.image":  "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}
DUPLICATE_FOR_CAMERA3 = "observation.images.image2"   # this one fills camera3 too

# ── Helpers ──────────────────────────────────────────────────────────────────

def remap_camera_keys(frame: dict) -> dict:
    """
    Translate LIBERO camera key names to the keys smolvla_base expects.
    
    LIBERO dataset:    observation.images.image, observation.images.image2
    smolvla_base:      observation.images.camera1/2/3

    Since LIBERO only has 2 cameras but the policy expects 3,
    we duplicate image2 into camera3. This is fine for a smoke-test —
    when we fine-tune on our own data we'd collect a real 3rd view.
    """
    # Grab the camera3 tensor before we start popping keys
    camera3_tensor = frame[DUPLICATE_FOR_CAMERA3]

    for old_key, new_key in CAMERA_KEY_MAP.items():
        frame[new_key] = frame.pop(old_key)

    frame["observation.images.camera3"] = camera3_tensor
    return frame


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_action_breakdown(action_np):
    """Print the shape and first predicted action step."""
    ndim = action_np.ndim

    if ndim == 3:
        b, c, a = action_np.shape
        print(f"  Shape: 3D — (batch={b}, chunk={c}, action_dim={a})")
        first_step = action_np[0, 0, :]
    elif ndim == 2:
        c, a = action_np.shape
        print(f"  Shape: 2D — (chunk={c}, action_dim={a})")
        first_step = action_np[0, :]
    else:
        print(f"  Shape: 1D — (action_dim={action_np.shape[0]})")
        first_step = action_np

    print(f"\n  First predicted step:")
    print(f"    Joints 0-5 : {first_step[:6]}")
    print(f"    Gripper    : {first_step[6]:.4f}")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("Phase 0.1 — SmolVLA Minimal Inference")
    print("=" * 70)

    # 1. Device
    device = select_device()
    print(f"Device: {device}")

    # 2. Load policy
    print(f"\nLoading policy: {MODEL_ID} ...")
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID).to(device).eval()

    # 3. Pre/post processors
    print("Building pre/post processors ...")
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # 4. Load exactly one episode from LIBERO
    #    episodes=[0]       → only fetches episode 0, not the full 1.5 GB corpus
    #    video_backend=pyav → uses PyAV instead of broken torchcodec on macOS ARM
    print(f"\nLoading episode 0 from {DATASET_ID} ...")
    dataset = LeRobotDataset(DATASET_ID, episodes=[0], video_backend="pyav")

    frame_index = dataset.meta.episodes["dataset_from_index"][0]
    frame = dict(dataset[frame_index])

    # 5. Remap camera keys: LIBERO names → smolvla_base names
    frame = remap_camera_keys(frame)

    # 6. Optional instruction override
    original_task = frame.get("task", "N/A")
    print(f"\nDataset instruction: '{original_task}'")
    if input("Override instruction? (y/n): ").strip().lower() == "y":
        frame["task"] = input("Enter instruction: ").strip()
        print(f"  → Using: '{frame['task']}'")
    else:
        print(f"  → Keeping original")

    # 7. Preprocess  (normalise images, tokenise task string, move to device)
    batch = preprocess(frame)

    # 8. Inference
    print("\nRunning inference ...")
    with torch.inference_mode():
        pred_action  = policy.select_action(batch)
        final_action = postprocess(pred_action)   # unnormalise back to real-world scale

    # 9. Verify output
    action_np       = final_action.cpu().numpy()
    is_valid_shape  = action_np.shape[-1] == 7

    print(f"\n✓ Inference complete!")
    print(f"  Raw tensor shape: {action_np.shape}")

    if is_valid_shape:
        print("  ✓ Action dim = 7  (6 joints + 1 gripper) — correct for Panda arm")
        print_action_breakdown(action_np)
    else:
        print(f"  ✗ Unexpected action dim: {action_np.shape[-1]} (expected 7)")

    # 10. Save results
    results = {
        "instruction":  frame["task"],
        "action":       action_np.tolist(),
        "action_shape": list(action_np.shape),
        "device":       str(device),
        "is_valid_shape": is_valid_shape,
    }
    out_path = OUTPUT_DIR / "phase0_1_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Results saved → {out_path}")
    print("\nPhase 0.1 complete!")


if __name__ == "__main__":
    main()