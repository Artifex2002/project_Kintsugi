"""
Optimizer & Scalability Mechanisms
==================================
Implements Block Decomposition and Farthest Point Clustering (FPC) 
to keep the Bayesian Optimization mathematically feasible and fast.
"""

import torch
import numpy as np
import random

def hamming_distance(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise Hamming distance (number of differing tokens) 
    between two batches of sequences.
    
    Args:
        X1 (torch.Tensor): Tensor of shape [N, L]
        X2 (torch.Tensor): Tensor of shape [M, L]
        
    Returns:
        torch.Tensor: Distance matrix of shape [N, M]
    """
    # X1.unsqueeze(1) -> [N, 1, L]
    # X2.unsqueeze(0) -> [1, M, L]
    # Indicator function (X1 != X2) summed over the sequence length L
    return (X1.unsqueeze(1) != X2.unsqueeze(0)).float().sum(dim=-1)

def farthest_point_clustering(history_X: torch.Tensor, history_Y: torch.Tensor, max_samples: int = 512):
    """
    Subset of Data (SoD) method with Farthest Point Clustering (FPC).
    Grabs the most diverse subset of past queries to prevent GP OOM errors.
    
    Args:
        history_X (torch.Tensor): All previously evaluated text integer arrays [H, L].
        history_Y (torch.Tensor): All previously evaluated MSE losses [H].
        max_samples (int): The maximum number of sequences to keep.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The subsampled X and Y tensors.
    """
    H = history_X.shape[0]
    
    # If history is small, just return it
    if H <= max_samples:
        return history_X, history_Y
    
    # 1. Start with a random sequence index
    sub_indices = [torch.randint(0, H, (1,)).item()]
    
    # Track the minimum distance from each point in history to our subsampled set
    min_distances = hamming_distance(history_X, history_X[sub_indices[0]:sub_indices[0]+1]).squeeze(1)
    
    # 2. Iteratively pick the sequence furthest away from everything chosen so far
    while len(sub_indices) < max_samples:
        farthest_idx = torch.argmax(min_distances).item()
        sub_indices.append(farthest_idx)
        
        # Update distances
        new_distances = hamming_distance(history_X, history_X[farthest_idx:farthest_idx+1]).squeeze(1)
        min_distances = torch.minimum(min_distances, new_distances)
        
    sub_indices = torch.tensor(sub_indices, device=history_X.device)
    return history_X[sub_indices], history_Y[sub_indices]

class BlockDecomposer:
    """
    Chunks the text sequence into smaller blocks and scores their vulnerability.
    """
    def __init__(self, sequence_length: int, block_size: int = 4):
        self.sequence_length = sequence_length
        self.block_size = block_size
        self.blocks = self._create_blocks()
        
    def _create_blocks(self) -> list:
        """Divides the sequence indices into disjoint blocks."""
        blocks = []
        for i in range(0, self.sequence_length, self.block_size):
            end = min(i + self.block_size, self.sequence_length)
            blocks.append(list(range(i, end)))
        return blocks
        
    def score_blocks(self, betas: torch.Tensor) -> list:
        """
        Scores each block based on the learned ARD parameters (beta).
        Smaller beta = higher importance.
        Score = sum(1 / beta_i) for i in block.
        """
        scores = []
        for block in self.blocks:
            # We take the inverse of beta because a smaller length-scale (beta) 
            # implies the token causes drastic shifts in the output loss.
            score = sum(1.0 / betas[i].item() for i in block)
            scores.append(score)
        return scores
        
    def get_most_important_block(self, betas: torch.Tensor) -> list:
        """Returns the list of indices for the most critical block."""
        # If the GP hasn't been fit yet (no betas), pick a random block
        if betas is None:
            return random.choice(self.blocks)
            
        scores = self.score_blocks(betas)
        best_idx = np.argmax(scores)
        return self.blocks[best_idx]

# ─── Sanity Check ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 1. Test Farthest Point Clustering
    print("Testing Farthest Point Clustering...")
    mock_X = torch.randint(0, 5, (1000, 16)) # 1000 history queries
    mock_Y = torch.randn(1000)
    
    sub_X, sub_Y = farthest_point_clustering(mock_X, mock_Y, max_samples=100)
    print(f"Original History: {mock_X.shape[0]} | Subsampled History: {sub_X.shape[0]}")
    
    # 2. Test Block Decomposition
    print("\nTesting Block Decomposition...")
    decomposer = BlockDecomposer(sequence_length=16, block_size=4)
    print(f"Created {len(decomposer.blocks)} blocks: {decomposer.blocks}")
    
    # Mock some learned beta values from the GP
    # Let's pretend the last 4 tokens (our adversarial suffix!) have tiny betas (high impact)
    mock_betas = torch.tensor([5.0]*12 + [0.1]*4) 
    scores = decomposer.score_blocks(mock_betas)
    
    print(f"\nMock Betas: {mock_betas.numpy()}")
    print(f"Block Scores: {np.round(scores, 2)}")
    print(f"Targeting Block: {decomposer.get_most_important_block(mock_betas)}")