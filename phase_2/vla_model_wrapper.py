"""
VLA Model Wrapper
==========================
This module abstracts the HuggingFaceVLA/smolvla_libero model into a clean 
black-box interface for our Black Box attack loop.
"""

import torch
import torch.nn.functional as F
import numpy as np
import copy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import torchvision.transforms.functional as TF

class VLAModel: # Renamed to VLAModel to match vla_attack.py imports
    """
    A black-box wrapper for HuggingFaceVLA/smolvla_libero.
    Provides methods to query the model and compute the adversarial loss over an image batch.
    """
    def __init__(self, model_id: str = "HuggingFaceVLA/smolvla_libero", device: str = None):
        self.model_id = model_id
        
        # Determine device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        print(f"Initializing VLA Wrapper on {self.device}...")

        # Load policy
        self.policy = SmolVLAPolicy.from_pretrained(self.model_id).to(self.device).eval()
        
        # Load processors
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            self.model_id,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        print("✓ Policy and Processors loaded successfully.")

    def get_action(self, text_instruction: str, image_context: dict) -> torch.Tensor:
        """
        Passes the text and single image context to the VLA and returns the continuous action array.
        """
        frame = copy.deepcopy(image_context)
        frame["task"] = text_instruction
        
        # Preprocess into a model-ready batch
        batch = self.preprocess(frame)
        
        with torch.inference_mode():
            pred_action = self.policy.select_action(batch)
            final_action = self.postprocess(pred_action)
            
        return final_action

    def compute_loss(self, text_instruction: str, target_action: np.ndarray, image_batch: list,
                     lam: float = 0.3, eps: float = 0.05, anchor_weight: float = 0.2) -> float:
        """
        Computes the target objective function for the Bayesian Optimizer using
        a normalized Hybrid Cosine + Anchored MSE loss, with discrete gripper handling.
        """
        # Convert target_action to tensor and ensure it's squeezed to a 1D vector
        if isinstance(target_action, np.ndarray):
            target_tensor = torch.tensor(target_action, dtype=torch.float32, device=self.device)
        else:
            target_tensor = target_action.clone().detach().to(self.device)
        target_tensor = target_tensor.squeeze()
            
        # 1. Setup Adaptive & Anchor Weights for MSE
        active_mask = target_tensor.abs() > eps
        active_mags = target_tensor.abs() * active_mask.float()
        norm = active_mags.sum().clamp(min=1e-6)
        
        # Give active joints proportional weights, give inactive joints strict anchor weight
        weights = torch.where(active_mask, active_mags / norm, 
                              torch.full_like(target_tensor, anchor_weight))
                              
        # --- FIX 1: The Gripper Override ---
        # The gripper (index 6) is a binary actuator, not a continuous spatial joint. 
        # We hardcode its weight to prevent it from dominating the adaptive magnitude norm.
        weights[6] = 0.05
        
        total_loss = 0.0
        
        # Iterate over our diverse subset of images
        for img in image_batch:
            img_tensor = TF.to_tensor(img)
            
            frame_context = {
                "observation.images.image": img_tensor,
                "observation.state": torch.tensor(
                    [-0.053380046, 0.007029631, 0.678328096, 3.140769243, 0.001759327, -0.089944183, 0.038788661, -0.038787212], 
                    dtype=torch.float32
                )
            }
            
            # Get output action
            current_action = self.get_action(text_instruction, frame_context)
            current_action = current_action.to(self.device).squeeze()
            
            # Term 1: Anchored Adaptive MSE (All 7 joints)
            # --- FIX 2: Scale Normalization ---
            # We divide by weights.sum() to prevent early massive errors from drowning out the cosine signal
            mse_term = (weights * (target_tensor - current_action).pow(2)).sum() / weights.sum()
            
            # Term 2: Arm-Only Cosine Similarity (Indices 0 to 5, ignoring gripper at 6)
            p_arm = current_action[:6]
            t_arm = target_tensor[:6]
            
            # Minimize (1 - cos_sim) to maximize cosine similarity
            cos_sim = F.cosine_similarity(p_arm.unsqueeze(0), t_arm.unsqueeze(0), eps=1e-6)
            orient_loss = (1 - cos_sim.squeeze())
            
            # Combine terms using lambda weighting
            step_loss = (lam * orient_loss) + ((1 - lam) * mse_term)
            total_loss += step_loss.item()
            
        # Average across all images in the batch
        avg_loss = total_loss / len(image_batch)
        
        # Return negative loss (Bayesian Optimization seeks to MAXIMIZE)
        return -avg_loss

# ── Quick Sanity Check (Runs only if executed directly) ──────────────────────
if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    
    # 1. Initialize wrapper
    wrapper = VLAModel(model_id="HuggingFaceVLA/smolvla_libero")
    
    # 2. Create a mock list of dummy PIL Images (Black squares)
    # In reality, this simulates loading 3 images from your dataset
    dummy_image_batch = [Image.new('RGB', (224, 224), color = 'black') for _ in range(3)]
    
    # 3. Test compute_loss with a batch of images and a dummy numpy target
    test_text = "move the robot arm forward"
    dummy_target = np.zeros(7) # 7-DOF target action
    
    print("\nRunning Image-Invariant Loss Computation...")
    loss = wrapper.compute_loss(test_text, dummy_target, dummy_image_batch)
    print(f"Calculated Average Loss (Neg MSE): {loss:.6f}")