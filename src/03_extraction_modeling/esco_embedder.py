import os
import torch
from sentence_transformers import SentenceTransformer, util

class ESCOEmbedder:
    """
    Generates dense vector embeddings using transformer models to capture 
    semantic relationships between candidate experiences and job roles, 
    mapping them to the standardized ESCO taxonomy.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # A highly efficient sentence-transformer model for generating semantic embeddings
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.embedder = SentenceTransformer(model_name, device=self.device)
        print(f"Embedding model loaded on: {self.device}")
        
        # In a full run, this would be populated from data/esco_taxonomy/esco_skills.csv
        self.esco_standard_terms = [
            "adaptive leadership", "software development", 
            "financial auditing", "ethical judgment", "team collaboration"
        ]
        
        # Pre-compute embeddings for the ESCO standard terms
        self.esco_embeddings = self.embedder.encode(self.esco_standard_terms, convert_to_tensor=True)

    def map_to_esco(self, extracted_skill: str):
        """
        Takes an extracted string (from RoBERTa or the Generative model),
        embeds it, and calculates the cosine similarity against the ESCO standard.
        """
        # Generate dense vector for the applicant's extracted skill
        skill_embedding = self.embedder.encode(extracted_skill, convert_to_tensor=True)
        
        # Calculate cosine similarities against all ESCO terms
        cosine_scores = util.cos_sim(skill_embedding, self.esco_embeddings)[0]
        
        # Find the best match
        best_match_idx = torch.argmax(cosine_scores).item()
        best_match_score = cosine_scores[best_match_idx].item()
        mapped_term = self.esco_standard_terms[best_match_idx]
        
        return {
            "original_extraction": extracted_skill,
            "esco_mapped_term": mapped_term,
            "confidence_score": round(best_match_score, 4)
        }

if __name__ == "__main__":
    mapper = ESCOEmbedder()
    
    # Quick test to prove the semantic matching works (replacing rigid keyword matching)
    test_extraction = "collaborates well with diverse groups"
    result = mapper.map_to_esco(test_extraction)
    
    print("\n--- Semantic Mapping Test ---")
    print(f"Extracted Phase: '{result['original_extraction']}'")
    print(f"Mapped ESCO Standard: '{result['esco_mapped_term']}'")
    print(f"Semantic Confidence: {result['confidence_score']}")