"""
Phase 0.1 - SmolVLA Minimal Inference Script
========================================================

This script:
1. Loads SmolVLA-450M from HuggingFace
2. Takes a text instruction + image from LIBERO
3. Gets action vector output
4. Confirms output shape for 7-DOF robot arm
5. Saves reference image for fixed I throughout PoC
6. Checks determinism (runs 3x, verifies whether identical outputs or not)

Reference Docs: 
https://huggingface.co/lerobot/smolvla_base
https://huggingface.co/docs/lerobot/introduction_processors
https://huggingface.co/docs/lerobot/env_processor
"""

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_ID   = "lerobot/smolvla_base"
DATASET_ID = "lerobot/libero"
OUTPUT_DIR = Path("phase0_outputs")

# smolvla_base was trained with these 3 camera keys.
# LIBERO only has 2 cameras, so we map image→camera1, image2→camera2,
# and duplicate image2 into camera3 as a placeholder.
CAMERA_KEY_MAP = {
    "observation.images.image":  "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}
DUPLICATE_FOR_CAMERA3 = "observation.images.image2"

# ── Helpers ──────────────────────────────────────────────────────────────────

def remap_camera_keys(frame: dict) -> dict:
    """
    Translate LIBERO camera key names to the keys smolvla_base expects.
    
    LIBERO dataset:    observation.images.image, observation.images.image2
    smolvla_base:      observation.images.camera1/2/3

    Since LIBERO only has 2 cameras but the policy expects 3,
    we duplicate image2 into camera3. This should be fine for a smoke-test.
    """
    camera3_tensor = frame[DUPLICATE_FOR_CAMERA3]

    for old_key, new_key in CAMERA_KEY_MAP.items():
        frame[new_key] = frame.pop(old_key)

    frame["observation.images.camera3"] = camera3_tensor
    return frame


def select_device() -> torch.device:
    """Select best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_reference_image(frame: dict, output_dir: Path) -> Path:
    """
    Save the reference image as fixed I for all PoC experiments.
    
    This image becomes our experimental control:
    - All text perturbations will be tested with THIS EXACT IMAGE
    - Ensures any action changes are due to text, not visual input
    
    Returns:
        Path to saved image
    """
    print("\n" + "="*70)
    print("Saving Reference Image (Fixed I)")
    print("="*70)
    
    # Use camera1 as the reference
    camera1_tensor = frame["observation.images.camera1"]
    
    # Convert tensor to PIL Image
    # Tensor shape: (C, H, W) in [0, 1] range
    img_np = camera1_tensor.cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    img_np = np.transpose(img_np, (1, 2, 0))  # (C,H,W) → (H,W,C)
    
    img_pil = Image.fromarray(img_np)
    
    # Save
    img_path = output_dir / "reference_image.png"
    img_pil.save(img_path)
    
    print(f"✓ Saved reference image: {img_path}")
    print(f"  Resolution: {img_np.shape[1]}x{img_np.shape[0]} (WxH)")
    print(f"  This image will be used as fixed I for all PoC experiments")
    
    return img_path


def check_determinism(policy, preprocess, postprocess, frame: dict) -> bool:
    """
    Verify SmolVLA is deterministic by running the same input 3 times.
    
    SmolVLA is autoregressive (not diffusion), so outputs should be identical.
    This is critical for attack evaluation - non-deterministic outputs would
    make it impossible to tell if your attack worked or the model just randomly
    changed its output.
    
    Returns:
        True if all 3 runs produce identical outputs, False otherwise
    """
    print("\n" + "="*70)
    print("Checking Determinism")
    print("="*70)
    print("Running identical input 3 times...")
    
    actions = []
    
    for run in range(3):
        # Create fresh batch each time
        batch = preprocess(frame)
        
        # Inference
        with torch.inference_mode():
            pred_action = policy.select_action(batch)
            final_action = postprocess(pred_action)
        
        action_np = final_action.cpu().numpy().flatten()[:7]
        actions.append(action_np)
        
        print(f"  Run {run+1}: {action_np}")
    
    # Compare all runs
    all_identical = True
    max_diff = 0.0
    
    for i in range(1, 3):
        if not np.allclose(actions[0], actions[i], atol=1e-6):
            all_identical = False
            diff = np.abs(actions[0] - actions[i]).max()
            max_diff = max(max_diff, diff)
    
    if all_identical:
        print("\n  ✓ All outputs IDENTICAL - SmolVLA is deterministic!")
        print("    → Attack evaluation will be reliable")
    else:
        print(f"\n  Outputs differ by up to {max_diff:.2e}")
        print("    This is likely due to:")
        print("    - VLA Model being stochastic in nature.")
        print("    - Floating point precision (acceptable if < 1e-5)")
        print("    - Non-deterministic GPU operations")
        
        if max_diff < 1e-5:
            print("    → Difference is negligible, treating as deterministic")
            all_identical = True
    
    return all_identical


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

    print(f"\n  Action Vector:")
    print(f"    Joints 0-5 : {first_step[:6]}")
    print(f"    Gripper    : {first_step[6]:.4f}")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("Phase 0.1 — SmolVLA Minimal Inference (Complete)")
    print("=" * 70)

    # 1. Device
    device = select_device()
    print(f"Device: {device}")

    # 2. Load policy
    print(f"\nLoading policy: {MODEL_ID} ...")
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID).to(device).eval()
    print("✓ Policy loaded")

    # 3. Pre/post processors
    print("\nBuilding pre/post processors ...")
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print("✓ Processors created")

    # 4. Load exactly one episode from LIBERO
    print(f"\nLoading episode 0 from {DATASET_ID} ...")
    dataset = LeRobotDataset(DATASET_ID, episodes=[0], video_backend="pyav")

    frame_index = dataset.meta.episodes["dataset_from_index"][0]
    frame = dict(dataset[frame_index])
    print("✓ Episode loaded")

    # 5. Remap camera keys
    frame = remap_camera_keys(frame)

    # 6. Save reference image (fixed I)
    ref_img_path = save_reference_image(frame, OUTPUT_DIR)

    # 7. Optional instruction override
    original_task = frame.get("task", "N/A")
    print("\n" + "="*70)
    print("Instruction")
    print("="*70)
    print(f"Dataset instruction: '{original_task}'")
    
    if input("Override instruction? (y/n): ").strip().lower() == "y":
        frame["task"] = input("Enter instruction: ").strip()
        print(f"  → Using: '{frame['task']}'")
    else:
        print(f"  → Keeping: '{original_task}'")

    # 8. Check determinism
    is_deterministic = check_determinism(policy, preprocess, postprocess, frame)

    # 9. Main inference run
    print("\n" + "="*70)
    print("Main Inference")
    print("="*70)
    
    batch = preprocess(frame)
    
    print("Running inference ...")
    with torch.inference_mode():
        pred_action  = policy.select_action(batch)
        final_action = postprocess(pred_action)

    # 10. Verify output
    action_np = final_action.cpu().numpy()
    is_valid_shape = action_np.shape[-1] == 7

    print(f"✓ Inference complete!")
    print(f"  Output shape: {action_np.shape}")

    if is_valid_shape:
        print("  ✓ Action dim = 7 (6 joints + 1 gripper) — Correct!")
        print_action_breakdown(action_np)
    else:
        print(f"  ✗ Unexpected action dim: {action_np.shape[-1]} (expected 7), no worries though, the inference pipeline works!")

    # 11. Save results
    results = {
        "instruction": frame["task"],
        "action": action_np.tolist(),
        "action_shape": list(action_np.shape),
        "device": str(device),
        "is_valid_shape": is_valid_shape,
        "is_deterministic": is_deterministic,
        "reference_image_path": str(ref_img_path),
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
    }
    
    out_path = OUTPUT_DIR / "phase0_1_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Results saved → {out_path}")

    # 12. Phase 0.1 Exit Criteria
    print("\n" + "="*70)
    print("Phase 0.1 Exit Criteria")
    print("="*70)
    
    exit_criteria = {
        "Can run smolvla.predict(instruction, image)": True,
        "Gets stable (7,) action vector": is_valid_shape,
        "Outputs are deterministic": is_deterministic,
        "Reference image saved": ref_img_path.exists(),
    }
    
    for criterion, passed in exit_criteria.items():
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {criterion}")
    
    all_passed = all(exit_criteria.values())
    
    if all_passed:
        print("\n Phase 0.1 COMPLETE!")
        print("\nNext Steps:")
        print("  → Phase 0.2: Define 3 target actions (easy, medium, hard)")
        print("  → Use reference_image.png as fixed I for all experiments")
    else:
        print("\n Some criteria not met - review above")
    
    print("="*70)


if __name__ == "__main__":
    main()