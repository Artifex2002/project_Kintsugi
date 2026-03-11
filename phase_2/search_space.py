"""
Hybrid Search Space Definition
==============================
Defines the discrete combinatorial search space for the Bayesian Optimizer.
Combines prefix synonym substitutions with unconstrained adversarial suffixes.
"""

import string
import random
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords

# Ensure required NLTK data is downloaded (runs once)
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading required NLTK datasets...")
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')

class HybridSearchSpace:
    def __init__(self, base_text: str, num_suffixes: int = 4, max_synonyms: int = 8, suffix_vocab_size: int = 500):
        """
        Initializes the search space.
        
        Args:
            base_text (str): The starting instruction.
            num_suffixes (int): Number of blank tokens to append at the end.
            max_synonyms (int): Max number of synonyms to allow per prefix word.
            suffix_vocab_size (int): The size of the vocabulary for the suffix tokens.
        """
        self.base_text = base_text
        self.num_suffixes = num_suffixes
        self.max_synonyms = max_synonyms
        self.suffix_vocab_size = suffix_vocab_size
        
        self.stop_words = set(stopwords.words('english'))
        
        # 1. Parse the base text (remove punctuation to make wordnet matching easier)
        clean_text = base_text.translate(str.maketrans('', '', string.punctuation))
        self.base_tokens = clean_text.strip().split()
        
        # 2. Build the suffix vocabulary (e.g., top common English words)
        self.suffix_vocab = self._build_suffix_vocabulary(suffix_vocab_size)
        
        # 3. Construct the full search space candidates
        self.candidates = []
        self._build_space()
        
        # 4. Store the bounds (how many choices exist for each token position)
        self.bounds = [len(c) for c in self.candidates]
        self.sequence_length = len(self.bounds)

    def _get_synonyms(self, word: str) -> list:
        """Finds synonyms for a given word using WordNet."""
        if word.lower() in self.stop_words:
            return [word] # Don't swap stop words (on, the, to, etc.)
            
        synonyms = set([word]) # Always include the original word
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                candidate = lemma.name().replace('_', ' ').lower()
                # Keep it strictly single-word to avoid messing up sequence length
                if ' ' not in candidate: 
                    synonyms.add(candidate)
                    
        # Sort for reproducibility, then truncate to max_synonyms
        synonyms = sorted(list(synonyms))
        return synonyms[:self.max_synonyms]

    def _build_suffix_vocabulary(self, vocab_size: int) -> list:
        """
        Creates a list of words for the adversarial suffix. 
        In a real scenario, you could load the 500 most common words from the Libero dataset.
        Here we use a diverse subset of WordNet lemmas as a proxy.
        """
        all_lemmas = list(wordnet.all_lemma_names())
        # Filter for basic alphabetic words, no numbers/symbols
        valid_words = [w for w in all_lemmas if w.isalpha() and len(w) > 2]
        
        # Use a fixed random seed to ensure the vocabulary is the same every time we run
        random.seed(42) 
        vocab = random.sample(valid_words, vocab_size)
        return sorted(vocab)

    def _build_space(self):
        """Assembles the candidate lists for both prefix and suffix."""
        # A. Prefix Search Space (Synonyms)
        for token in self.base_tokens:
            token_candidates = self._get_synonyms(token)
            self.candidates.append(token_candidates)
            
        # B. Suffix Search Space (Full Vocabulary)
        for _ in range(self.num_suffixes):
            self.candidates.append(self.suffix_vocab)

    def decode(self, indices: list) -> str:
        """
        Converts an array of selected indices back into a human-readable text string.
        
        Args:
            indices (list): A list of integers where indices[i] < self.bounds[i].
        Returns:
            str: The decoded adversarial sentence.
        """
        if len(indices) != self.sequence_length:
            raise ValueError(f"Expected {self.sequence_length} indices, got {len(indices)}")
            
        words = []
        for i, idx in enumerate(indices):
            # Clamp index just in case the optimizer oversteps
            safe_idx = int(max(0, min(idx, self.bounds[i] - 1)))
            words.append(self.candidates[i][safe_idx])
            
        return " ".join(words)

    def get_original_indices(self) -> list:
        """Returns the indices that reconstruct the exact original prompt (with dummy suffixes)."""
        indices = []
        # Original words are always at some index in the synonym list (usually 0 if we sorted, but we must find it)
        for i in range(len(self.base_tokens)):
            orig_word = self.base_tokens[i].lower()
            # Find where the original word ended up in the sorted synonym list
            try:
                idx = self.candidates[i].index(orig_word)
            except ValueError:
                idx = 0
            indices.append(idx)
            
        # For suffixes, just pick index 0 as a default baseline
        for _ in range(self.num_suffixes):
            indices.append(0)
            
        return indices

# ── Quick Sanity Check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    starter_string = "Grasp blue box on counter, put it on burner, rotate the knob to turn off and end the task."
    
    # Initialize Search Space
    print("Building Hybrid Search Space...")
    space = HybridSearchSpace(base_text=starter_string, num_suffixes=4, max_synonyms=8, suffix_vocab_size=500)
    
    print(f"\nBase tokens ({len(space.base_tokens)}): {space.base_tokens}")
    print(f"Total Sequence Length (Prefix + Suffix): {space.sequence_length}\n")
    
    print("Bounds (Number of choices per token position):")
    print(space.bounds)
    
    # Test Decoding the Original String
    orig_indices = space.get_original_indices()
    print(f"\nOriginal Indices: {orig_indices}")
    print(f"Decoded Original: {space.decode(orig_indices)}")
    
    # Test Decoding a Random Adversarial Perturbation
    import numpy as np
    random_indices = [np.random.randint(0, b) for b in space.bounds]
    print(f"\nRandom Adversarial Indices: {random_indices}")
    print(f"Decoded Adversarial: {space.decode(random_indices)}")