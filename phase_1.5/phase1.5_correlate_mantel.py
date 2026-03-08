import numpy as np
from skbio.stats.distance import mantel

# --- 1. Load Matrices ---
text_dists = np.load('text_distances.npy')
action_dists = np.load('action_distances.npy')

# --- 2. Calculate Mantel Test ---
print("Running Mantel permutations (this may take a moment)...")
rho, p_value, _ = mantel(
    text_dists, 
    action_dists, 
    method='spearman', 
    permutations=999
)

print(f"Mantel Correlation (ρ): {rho:.4f}")
print(f"Rigorous P-value: {p_value:.4f}")