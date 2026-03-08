import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

# --- Path Setup ---
output_dir = Path(__file__).parent / "phase1.5_outputs"
text_matrix_path = output_dir / 'text_distances.npy'
action_matrix_path = output_dir / 'action_distances.npy'

# --- 1. Load Matrices ---
text_dists = np.load(text_matrix_path)
action_dists = np.load(action_matrix_path)

# --- 2. Extract Upper Triangle ---
# k=1 ignores the diagonal (distance of 0 to itself)
upper_tri_indices = np.triu_indices(text_dists.shape[0], k=1)

text_flat = text_dists[upper_tri_indices]
action_flat = action_dists[upper_tri_indices]

# --- 3. Calculate Correlation ---
rho, p_value = spearmanr(text_flat, action_flat)

print(f"Standard Spearman Correlation (ρ): {rho:.4f}")
print("Interpretation:")
if rho > 0.3:
    print(" Strong signal: Proceed!")
elif rho > 0.15:
    print(" Weak signal: Proceed with caution.")
else:
    print(" No signal: Abort.")