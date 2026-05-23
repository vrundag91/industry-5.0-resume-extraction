import os
import re

class TextPreprocessor:
    """
    Cleans and standardizes anonymized resume text.
    Removes irrelevant formatting, special characters, and reduces lexical noise
    to prepare high-fidelity data for human annotation and model training.
    """
    def __init__(self):
        # Basic set of common stop words to reduce noise
        self.stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "but", "by",
            "for", "if", "in", "into", "is", "it",
            "no", "not", "of", "on", "or", "such",
            "that", "the", "their", "then", "there", "these",
            "they", "this", "to", "was", "will", "with"
        }

    def clean_text(self, text: str) -> str:
        """Applies normalization and noise reduction to a single string."""
        # Convert to lowercase for normalization
        text = text.lower()
        
        # Remove formatting tags (e.g., if HTML snuck in from scraping)
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove special characters and punctuation (keep alphanumeric and spaces)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stop words
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        
        return ' '.join(filtered_words)

    def process_directory(self, input_dir: str, output_dir: str):
        """Processes all anonymized files and preps them for human annotation."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in os.listdir(input_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                
                cleaned_content = self.clean_text(content)
                
                with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                print(f"Cleaned and normalized: {filename}")

if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    # Define absolute paths based on the current script location
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    anon_dir = os.path.join(base_dir, "data", "02_anonymized")
    
    # We save this to a subfolder in 03_annotated as the "baseline" text to be labeled
    clean_dir = os.path.join(base_dir, "data", "03_annotated", "00_preprocessed_baseline")
    
    preprocessor.process_directory(input_dir=anon_dir, output_dir=clean_dir)
    print("Text preprocessing and noise reduction complete.")