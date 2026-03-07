"""
Phase 1 - Process Embeddings & Actions (Streamlined)
========================================================

This script:
1. Loads SmolVLA (Libero) and its official pre/post-processors
2. Loads our pre-saved reference image
3. Loads the generated prompts JSON
4. Iterates through every prompt:
   - Calculates the text embedding
   - Injects the prompt + fixed image into the VLA
   - Calculates the continuous action vector
5. Saves the final correlated dataset
"""

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Import our custom embedder from Step 1
from phase1_embedder import get_embedding

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_ID   = "HuggingFaceVLA/smolvla_libero"  # Explicitly using the Libero finetune for our experiments
INPUT_JSON = Path("phase1_outputs/generated_prompts.json")
OUTPUT_DIR = Path("phase1_outputs")
REFERENCE_IMAGE_PATH = Path("../phase_0/phase0_outputs/reference_image.png")

# ── Helpers ──────────────────────────────────────────────────────────────────

def select_device() -> torch.device:
    """Select best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_local_reference_image(image_path: Path) -> torch.Tensor:
    """
    Loads the local PNG image and converts it to a standard PyTorch tensor 
    [C, H, W] in the range [0.0, 1.0]. The LeRobot preprocessor will handle 
    the rest of the normalization (like resizing or standard deviation).
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Could not find reference image at {image_path}")
        
    print(f"\nLoading fixed reference image from: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    # Convert to standard tensor format
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(image)
    
    return img_tensor

def auto_populate_inputs(img_tensor: torch.Tensor, policy_config) -> dict:
    """
    Dynamically checks what keys the policy expects (images and state)
    and populates them to create a valid base frame.
    """
    frame = {}
    
    # Look at the model config using the correct attribute: 'input_features'
    expected_features = policy_config.input_features
    
    image_keys = []
    state_keys = []
    
    # expected_features contains PolicyFeature objects, not dicts
    for k, v in expected_features.items():
        # Convert the type (which might be an Enum) to a string to check safely
        feat_type_str = str(getattr(v, "type", ""))
        
        if "VISUAL" in feat_type_str:
            image_keys.append(k)
        elif "STATE" in feat_type_str:
            state_keys.append(k)
    
    print(f"Policy expects these image keys: {image_keys}")
    print(f"Policy expects these state keys: {state_keys}")
    print("Populating all required inputs with fixed reference data...")
    
    # 1. Inject the reference image into all expected camera inputs
    for key in image_keys:
        frame[key] = img_tensor
        
    # 2. Inject a REAL state vector for all expected state inputs
    real_libero_state = torch.tensor(
        [
        -0.05338004603981972,
        0.007029631175100803,
        0.6783280968666077,
        3.1407692432403564,
        0.0017593271331861615,
        -0.08994418382644653,
        0.03878866136074066,
        -0.03878721222281456
        ], 
        dtype=torch.float32
    )
    
    for key in state_keys:
        # We assign the tensor directly without querying shape to avoid another dataclass error
        frame[key] = real_libero_state
        
    return frame

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 1 — Batch Processing Embeddings & Actions (Local Image)")
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

    # 4. Load local image & map to expected camera keys
    base_img_tensor = load_local_reference_image(REFERENCE_IMAGE_PATH)
    base_frame = auto_populate_inputs(base_img_tensor, policy.config)

    # 5. Load generated prompts
    if not INPUT_JSON.exists():
        print(f"\n✗ Error: {INPUT_JSON} not found. Run the phase1_prompt_generator script first.")
        return
        
    with open(INPUT_JSON, "r") as f:
        categories = json.load(f)

    print("\n" + "="*70)
    print("Beginning Batch Processing")
    print("="*70)

    results = {}

    with torch.inference_mode():
        for category_name, texts in categories.items():
            print(f"\nProcessing {category_name.upper()} ({len(texts)} prompts) ...")
            results[category_name] = []
            
            for text in texts:
                # 1. Text Embedding
                emb_tensor = get_embedding(text)
                
                # 2. Inject text into our base frame
                current_frame = base_frame.copy()
                current_frame["task"] = text
                
                # 3. Preprocess to format the batch
                batch = preprocess(current_frame)
                
                # 4. VLA Inference
                pred_action = policy.select_action(batch)
                final_action = postprocess(pred_action)
                
                # LeRobot returns (batch_size, action_dim) 
                action_np = final_action.cpu().numpy()
                
                # We have a batch size of 1, so we slice Batch 0 to get our 1D 7-DOF array
                first_action = action_np[0] 
                
                results[category_name].append({
                    "text": text,
                    "embedding": emb_tensor.cpu().numpy().tolist(),
                    "action": first_action.tolist()
                })
            
            print(f"  ✓ Processed {len(texts)} items")

    # 6. Save Final Results
    out_path = OUTPUT_DIR / "phase1_prompt-embedding-action_dataset.json"
    out_path.write_text(json.dumps(results, indent=2))
    
    print("\n" + "="*70)
    print(f"✓ Success! Data saved to: {out_path}")
    print("="*70)

if __name__ == "__main__":
    main()