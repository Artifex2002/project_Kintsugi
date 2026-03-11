"""
VLA Model Wrapper
==========================
This module abstracts the HuggingFaceVLA/smolvla_libero model into a clean 
black-box interface for our Black Box attack loop.
"""

import torch
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

    def compute_loss(self, text_instruction: str, target_action: np.ndarray, image_batch: list) -> float:
        """
        Computes the target objective function for the Bayesian Optimizer.
        Calculates the Expectation over the image distribution (E_I).
        
        Args:
            text_instruction (str): The text to evaluate.
            target_action (np.ndarray): The continuous action array we want the robot to output.
            image_batch (list): A list of PIL Images representing our visual diversity.
            
        Returns:
            float: The negative average MSE loss across the batch.
        """
        # Convert target_action to tensor if it isn't already, and move to device
        if isinstance(target_action, np.ndarray):
            target_tensor = torch.tensor(target_action, dtype=torch.float32, device=self.device)
        else:
            target_tensor = target_action.to(self.device)
            
        total_mse = 0.0
        
        # Iterate over our diverse subset of images
        for img in image_batch:
            # Convert PIL Image to PyTorch Tensor [C, H, W] in range [0, 1]
            img_tensor = TF.to_tensor(img)
            
            frame_context = {
                "observation.images.image": img_tensor,
                # Injecting a real, valid arm state pulled from the Libero dataset!
                "observation.state": torch.tensor(
                    [-0.053380046, 0.007029631, 0.678328096, 3.140769243, 0.001759327, -0.089944183, 0.038788661, -0.038787212], 
                    dtype=torch.float32
                )
            }
            # Get the actual output action from the model for this specific image
            current_action = self.get_action(text_instruction, frame_context)
            current_action = current_action.to(self.device)
            
            # 1. Defining the weights (6 joints at full weight, 1 gripper heavily penalized)
            # Ensure it is on the same device as our tensors
            action_weights = torch.tensor(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05], 
                dtype=torch.float32, 
                device=self.device
            )
            
            # 2. Squeeze current_action just in case the model returns a batched shape like [1, 7]
            current_action = current_action.squeeze()
            target_tensor = target_tensor.squeeze()
            
            # 3. Compute the Weighted Mean Squared Error
            squared_errors = (current_action - target_tensor) ** 2
            weighted_mse = torch.mean(squared_errors * action_weights)
            
            total_mse += weighted_mse.item()
            
        # Average the MSE across all images in the batch
        avg_mse = total_mse / len(image_batch)
        
        # Return negative MSE (Bayesian Optimization seeks to MAXIMIZE)
        return -avg_mse

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