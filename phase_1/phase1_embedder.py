import torch
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    """
    A singleton-style class to ensure the model is only loaded into memory once 
    per runtime, saving memory and compute time.
    """
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        # Auto-detect hardware acceleration (CUDA for Nvidia, MPS for Apple Silicon)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"Loading embedding model '{model_name}' on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed(self, text: str | list[str]) -> torch.Tensor:
        """
        Takes a string or list of strings and returns normalized L2 embeddings.
        """
        # The encode method handles tokenization and forwarding automatically.
        embeddings = self.model.encode(
            text,
            convert_to_tensor=True,      # Returns PyTorch tensors instead of NumPy arrays
            normalize_embeddings=True    # CRITICAL: Forces L2 norm = 1
        )
        return embeddings

# Initialize a global instance so it's ready to use on import
_global_embedder = None

def get_embedding(text: str | list[str]) -> torch.Tensor:
    """
    Main function to import into other files. 
    Usage:
        from embedder import get_embedding
        vec = get_embedding("pick up the apple")
    """
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = TextEmbedder()
    
    return _global_embedder.embed(text)

# --- Quick Test (runs only if this file is executed directly) ---
if __name__ == "__main__":
    text1 = "pick up the apple"
    text2 = "grab the red fruit"
    
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    
    # Verify dimensions (should be 768 for all-mpnet-base-v2)
    print(f"Embedding shape: {emb1.shape}")
    
    # Verify L2 norm is 1
    norm = torch.linalg.vector_norm(emb1)
    print(f"L2 Norm: {norm.item():.4f}")
    
    # Calculate Cosine Similarity (Since L2 norm=1, dot product == cosine similarity)
    similarity = torch.dot(emb1, emb2)
    print(f"Similarity between '{text1}' and '{text2}': {similarity.item():.4f}")