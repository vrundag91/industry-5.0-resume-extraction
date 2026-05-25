import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class FairnessAuditor:
    """
    Calculates specific quantitative fairness metrics across demographic groups 
    to actively identify and eliminate algorithmic bias.
    """
    def __init__(self, data: pd.DataFrame, protected_attribute: str, target_column: str, prediction_column: str):
        self.data = data
        self.protected_attribute = protected_attribute # e.g., 'gender' or 'age_group'
        self.y_true = data[target_column]
        self.y_pred = data[prediction_column]
        
    def _get_group_metrics(self, group_val):
        """Helper to get True Positives, False Positives, etc. for a specific group."""
        mask = self.data[self.protected_attribute] == group_val
        tn, fp, fn, tp = confusion_matrix(self.y_true[mask], self.y_pred[mask], labels=[0, 1]).ravel()
        return tn, fp, fn, tp

    def disparate_impact(self, privileged_grp, unprivileged_grp):
        """Ratio of positive prediction rates between unprivileged and privileged groups."""
        unpriv_pos_rate = self.y_pred[self.data[self.protected_attribute] == unprivileged_grp].mean()
        priv_pos_rate = self.y_pred[self.data[self.protected_attribute] == privileged_grp].mean()
        return unpriv_pos_rate / priv_pos_rate if priv_pos_rate > 0 else 0

    def statistical_parity_difference(self, privileged_grp, unprivileged_grp):
        """Absolute difference in selection rates."""
        unpriv_pos_rate = self.y_pred[self.data[self.protected_attribute] == unprivileged_grp].mean()
        priv_pos_rate = self.y_pred[self.data[self.protected_attribute] == privileged_grp].mean()
        return unpriv_pos_rate - priv_pos_rate

    def equal_opportunity_difference(self, privileged_grp, unprivileged_grp):
        """Difference in True Positive Rates (TPR)."""
        _, _, fn_u, tp_u = self._get_group_metrics(unprivileged_grp)
        _, _, fn_p, tp_p = self._get_group_metrics(privileged_grp)
        
        tpr_u = tp_u / (tp_u + fn_u) if (tp_u + fn_u) > 0 else 0
        tpr_p = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else 0
        return tpr_u - tpr_p

    def average_odds_difference(self, privileged_grp, unprivileged_grp):
        """Average of difference in False Positive Rates and True Positive Rates."""
        tn_u, fp_u, fn_u, tp_u = self._get_group_metrics(unprivileged_grp)
        tn_p, fp_p, fn_p, tp_p = self._get_group_metrics(privileged_grp)
        
        fpr_u = fp_u / (fp_u + tn_u) if (fp_u + tn_u) > 0 else 0
        fpr_p = fp_p / (fp_p + tn_p) if (fp_p + tn_p) > 0 else 0
        
        tpr_diff = self.equal_opportunity_difference(privileged_grp, unprivileged_grp)
        fpr_diff = fpr_u - fpr_p
        
        return 0.5 * (fpr_diff + tpr_diff)

    def theil_index(self):
        """Measures generalized entropy/inequality of benefit allocation."""
        # Simplified calculation for binary outcomes
        b = self.y_pred - self.y_true + 1 # benefit
        mean_b = np.mean(b)
        if mean_b == 0: return 0
        return np.mean((b / mean_b) * np.log(b / mean_b + 1e-9))

if __name__ == "__main__":
    import os
    print("Fairness Auditor initialized. Computing Industry 5.0 bias metrics...\n")
    
    # Load the synthetic fairness data
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "03_annotated", "synthetic_fairness_data.csv")
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        # Initialize the auditor targeting our synthetic columns
        auditor = FairnessAuditor(
            data=df, 
            protected_attribute="demographic_group", 
            target_column="true_qualification", 
            prediction_column="ai_prediction"
        )
        
        # We explicitly set Group_A as privileged and Group_B as unprivileged based on our synthetic logic
        priv_grp = "Group_A"
        unpriv_grp = "Group_B"
        
        print("--- INDUSTRY 5.0 FAIRNESS METRICS ---")
        print(f"Total Resumes Audited: {len(df)}")
        print(f"Statistical Parity Difference:  {auditor.statistical_parity_difference(priv_grp, unpriv_grp):.4f} (Ideal: 0.0)")
        print(f"Disparate Impact:               {auditor.disparate_impact(priv_grp, unpriv_grp):.4f} (Ideal: 1.0)")
        print(f"Equal Opportunity Difference:   {auditor.equal_opportunity_difference(priv_grp, unpriv_grp):.4f} (Ideal: 0.0)")
        print(f"Average Odds Difference:        {auditor.average_odds_difference(priv_grp, unpriv_grp):.4f} (Ideal: 0.0)")
        print(f"Theil Index (Inequality):       {auditor.theil_index():.4f} (Ideal: 0.0)")
        print("-------------------------------------")
    else:
        print("Error: synthetic_fairness_data.csv not found. Please run the synthetic annotator first.")