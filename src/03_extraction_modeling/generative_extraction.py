import os
import torch
from transformers import pipeline

class GenerativeExtractor:
    """
    Utilizes generative models (Llama/GPT architecture) to evaluate broader 
    semantic context and extract nuanced, transversal, and human-centric competencies.
    """
    def __init__(self, model_name="gpt2"): 
        # Note: Using gpt2 as a lightweight local default. 
        # For production, this can be swapped to "meta-llama/Llama-2-7b-chat-hf" or an OpenAI API call.
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize the text-generation pipeline
        # device=0 for CUDA/MPS, -1 for CPU
        self.extractor = pipeline(
            "text-generation", 
            model=model_name, 
            device=0 if torch.cuda.is_available() or torch.backends.mps.is_available() else -1
        )
        print(f"Generative Extractor initialized on: {self.device}")

    def prompt_model(self, text: str) -> str:
        """
        Constructs the prompt forcing the model to focus on human-centric skills
        from the Industry 5.0 perspective.
        """
        prompt = (
            f"Analyze the following resume text and extract ONLY the nuanced, "
            f"human-centric, and transversal competencies (e.g., emotional intelligence, "
            f"adaptability, ethical judgment). \n\nResume: {text}\n\nCompetencies:"
        )
        
        # Generate the response
        outputs = self.extractor(
            prompt, 
            max_new_tokens=50, 
            num_return_sequences=1,
            truncation=True
        )
        return outputs[0]['generated_text']

    def extract_competencies(self, text_list: list):
        """Processes a batch of preprocessed resumes."""
        results = []
        for text in text_list:
            competencies = self.prompt_model(text)
            results.append(competencies)
        return results

if __name__ == "__main__":
    gen_pipeline = GenerativeExtractor()
    print("Generative pipeline ready to evaluate broader semantic context.")