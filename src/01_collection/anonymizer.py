import re
import os

class ResumeAnonymizer:
    """
    Deterministic PII removal engine.
    Removes emails, phone numbers, and URLs to ensure strict data privacy
    compliance before any human annotation or model training occurs.
    """
    def __init__(self):
        # Updated Regex patterns for messy/flattened real-world data
        
        # Catches standard emails, AND flattened ones like "namegmailcom"
        self.email_pattern = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+|[a-zA-Z0-9]+(?:gmail|yahoo|hotmail|outlook)com)'
        
        # Catches standard phone numbers, and solid blocks of 10+ digits
        self.phone_pattern = r'(\(?\b[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b|[0-9]{10,12})'
        
        # Catches standard URLs, AND flattened ones like "httpwwwindeedcom..."
        self.url_pattern = r'(https?://[^\s]+|www\.[^\s]+|http[s]?[a-zA-Z0-9]+|www[a-zA-Z0-9]+)'
        
    def anonymize_text(self, text: str) -> str:
        """Replaces PII with standard placeholders."""
        text = re.sub(self.email_pattern, '[REDACTED_EMAIL]', text)
        text = re.sub(self.phone_pattern, '[REDACTED_PHONE]', text)
        text = re.sub(self.url_pattern, '[REDACTED_URL]', text)
        return text

    def process_directory(self, input_dir: str, output_dir: str):
        """Processes all text files in the raw directory and saves to anonymized."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in os.listdir(input_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                
                safe_content = self.anonymize_text(content)
                
                with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                    f.write(safe_content)
                print(f"Anonymized: {filename}")

if __name__ == "__main__":
    anonymizer = ResumeAnonymizer()
    
    # Define absolute paths based on the current script location
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(base_dir, "data", "01_raw_scraped")
    anon_dir = os.path.join(base_dir, "data", "02_anonymized")
    
    anonymizer.process_directory(input_dir=raw_dir, output_dir=anon_dir)
    print("Deterministic anonymization protocol complete.")