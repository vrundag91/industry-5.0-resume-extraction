import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_split

class RobertaNERPipeline:
    """
    Deep learning NER pipeline utilizing RoBERTa.
    Targets explicit technical skills from preprocessed unstructured text.
    Configured for the 60/20/20 split defined in the methodology.
    """
    def __init__(self, model_name="roberta-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, add_prefix_space=True)
        
        # Hardware acceleration: automatically utilize Apple Silicon (MPS) or NVIDIA (CUDA)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Hardware Acceleration configured to use: {self.device}")
        
        # We will initialize the model with a placeholder for labels, to be updated dynamically
        self.model = None 

    def split_dataset(self, data):
        """
        Implements the strict 60/20/20 data split defined in the methodology:
        60% Training, 20% Validation, 20% Unbiased Testing.
        """
        # First split: 80% for train+val, 20% for pure test
        train_val_data, test_data = train_test_split(data, test_size=0.20, random_state=42)
        
        # Second split: 75% of the 80% (which equals 60% of total) for train, 25% of 80% (20% total) for val
        train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)
        
        return train_data, val_data, test_data

    def configure_training(self, output_dir: str):
        """
        Configures hyper-parameters including batch-size, learning rate, and epochs.
        """
        return TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,          # Standard learning rate for fine-tuning RoBERTa
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,          # Can be adjusted based on early stopping
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, 'logs'),
            use_mps_device=torch.backends.mps.is_available(), # Explicitly enable Mac GPU
        )

if __name__ == "__main__":
    pipeline = RobertaNERPipeline()
    
    # Define absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    annotated_dir = os.path.join(base_dir, "data", "03_annotated")
    model_output_dir = os.path.join(base_dir, "outputs", "roberta_ner_model")
    
    # Ensure output directory exists (since it's gitignored)
    os.makedirs(model_output_dir, exist_ok=True)
    
    print("RoBERTa NER Pipeline initialized. Ready to ingest 03_annotated data.")
    print("Awaiting high-fidelity labeled data arrays to execute Trainer().")