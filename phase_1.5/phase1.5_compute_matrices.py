import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

# --- Path Setup ---
# Automatically resolve paths relative to where this script lives
current_dir = Path(__file__).parent
project_root = current_dir.parent

input_file = project_root / "phase_1" / "phase1_outputs" / "phase1_prompt-embedding-action_dataset.json"
output_dir = current_dir / "phase1.5_outputs"

# Ensure output directory exists
output_dir.mkdir(exist_ok=True)

# --- 1. Load and Parse Dataset ---
print(f"Loading dataset from: {input_file}")
with open(input_file, 'r') as f:
    dataset = json.load(f)

# FIX 1: If the JSON was double-encoded as a string, parse it into an object
if isinstance(dataset, str):
    print("Detected double-encoded JSON string. Unpacking...")
    dataset = json.loads(dataset)

# FIX 2: If the list is wrapped in a parent dictionary, extract it
if isinstance(dataset, dict):
    print(f"Detected dictionary wrapper with keys: {list(dataset.keys())}")
    for key, val in dataset.items():
        if isinstance(val, list):
            print(f"Extracting the list of items from key: '{key}'")
            dataset = val
            break

# Extract lists and convert directly to numpy arrays
embeddings = np.array([item["embedding"] for item in dataset])
actions = np.array([item["action"] for item in dataset])

print(f"Loaded {len(dataset)} samples.")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Actions shape: {actions.shape}")

# --- 2. Compute Distances ---
print("Computing distance matrices...")
# Text -> Cosine distance
text_dist_matrix = squareform(pdist(embeddings, metric='cosine'))

# Actions -> L2 (Euclidean) distance
action_dist_matrix = squareform(pdist(actions, metric='euclidean'))

# --- 3. Save to Disk ---
np.save(output_dir / 'text_distances.npy', text_dist_matrix)
np.save(output_dir / 'action_distances.npy', action_dist_matrix)

print(f" Distance matrices computed and saved to: {output_dir}")