import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load Matrices & Flatten ---
text_dists = np.load('phase1.5_outputs/text_distances.npy')
action_dists = np.load('phase1.5_outputs/action_distances.npy')

upper_tri_indices = np.triu_indices(text_dists.shape[0], k=1)
text_flat = text_dists[upper_tri_indices]
action_flat = action_dists[upper_tri_indices]

# --- 2. Plotting ---
plt.figure(figsize=(10, 8))
plt.scatter(text_flat, action_flat, alpha=0.3, s=15, color='blue')

# Add a basic trend line
m, b = np.polyfit(text_flat, action_flat, 1)
plt.plot(text_flat, m*text_flat + b, color='red', linestyle='--', linewidth=2, label='Trend Line')

plt.title("Semantic Text Distance vs. Physical Action Distance")
plt.xlabel("Text Distance (Cosine)")
plt.ylabel("Action Distance (L2 Euclidean)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig('phase1.5_outputs/correlation_scatter.png', dpi=300)
print(" Scatter plot saved as 'correlation_scatter.png'")