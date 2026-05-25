import os
import json
import random
import pandas as pd

class SyntheticAnnotator:
    """
    Simulates human annotation to generate high-fidelity synthetic data.
    Creates structured NER labels for model training and demographic data 
    for fairness auditing.
    """
    def __init__(self):
        # Simulated Industry 5.0 transversal skills and technical skills
        self.tech_skills = ["python", "java", "kubernetes", "docker", "microbiology", "sql", "aws"]
        self.human_skills = ["adaptive", "ethical", "leadership", "collaboration", "empathy"]

    def generate_synthetic_data(self, input_dir: str, out_ner_path: str, out_fairness_path: str):
        ner_dataset = []
        fairness_records = []

        filenames = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        
        for idx, filename in enumerate(filenames):
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()

            # 1. Generate Synthetic NER Data (mimicking the 53 average labels per task)
            entities = []
            words = text.split()
            for w_idx, word in enumerate(words):
                if word in self.tech_skills:
                    entities.append({"word": word, "entity": "B-TECH_SKILL", "index": w_idx})
                elif word in self.human_skills:
                    entities.append({"word": word, "entity": "B-HUMAN_SKILL", "index": w_idx})
            
            ner_dataset.append({"id": idx, "text": text, "annotations": entities})

            # 2. Generate Synthetic Fairness Data
            # Simulating a scenario to test bias mitigation:
            # We inject a slight artificial bias where 'Group_A' has a higher prediction rate
            protected_group = random.choice(["Group_A", "Group_B"]) 
            
            # True label (1 = qualified, 0 = unqualified)
            true_label = random.choice([0, 1])
            
            # Predicted label (simulate the AI's decision)
            if protected_group == "Group_A":
                pred_label = 1 if random.random() > 0.3 else 0 # 70% chance of positive prediction
            else:
                pred_label = 1 if random.random() > 0.6 else 0 # 40% chance of positive prediction

            fairness_records.append({
                "resume_id": filename,
                "demographic_group": protected_group,
                "true_qualification": true_label,
                "ai_prediction": pred_label
            })

        # Save the NER dataset
        with open(out_ner_path, 'w', encoding='utf-8') as f:
            json.dump(ner_dataset, f, indent=4)
        print(f"Synthetic NER dataset generated at: {out_ner_path}")

        # Save the Fairness dataset
        df_fairness = pd.DataFrame(fairness_records)
        df_fairness.to_csv(out_fairness_path, index=False)
        print(f"Synthetic Fairness dataset generated at: {out_fairness_path}")

if __name__ == "__main__":
    annotator = SyntheticAnnotator()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    clean_dir = os.path.join(base_dir, "data", "03_annotated", "00_preprocessed_baseline")
    
    # Outputs go to the 03_annotated folder
    ner_out = os.path.join(base_dir, "data", "03_annotated", "synthetic_ner_data.json")
    fairness_out = os.path.join(base_dir, "data", "03_annotated", "synthetic_fairness_data.csv")
    
    annotator.generate_synthetic_data(clean_dir, ner_out, fairness_out)