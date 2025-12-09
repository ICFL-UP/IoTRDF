import joblib
import os
import sys
import pandas as pd
# Import necessary scikit-learn components for handling potential Pipelines and custom classes
from sklearn.base import BaseEstimator, TransformerMixin
# We use a try/except import for Pipeline as it might be nested
try:
    from sklearn.pipeline import Pipeline
except ImportError:
    # If Pipeline isn't available, we'll try to infer it later
    Pipeline = object

# --- Custom Placeholder Class ---
# This class MUST be defined for joblib.load() to succeed, as the model file 
# (likely a Pipeline) references it. It doesn't need to perform any logic here.
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        # Must accept any initialization arguments to match how the original was saved
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X

# --- Configuration ---
MODELS_DIR = "models"
POLICY_MODEL_PATH = os.path.join(MODELS_DIR, "policy", "model.joblib")
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "baseline", "model.joblib")

def inspect_model(model_path, model_name):
    """Loads a model and prints its structural characteristics, handling Pipelines."""
    print("=" * 60)
    print(f"--- INSPECTING MODEL: {model_name} ---")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}. Skipping.")
        return

    try:
        # Load the model using joblib
        model = joblib.load(model_path)
    except Exception as e:
        print(f"ERROR: Could not load the model from {model_path}. Please verify the file integrity.")
        print(f"Details: {e}")
        return

    # --- Pipeline Handling ---
    estimator = model
    if isinstance(model, Pipeline) or (hasattr(model, 'steps') and isinstance(model.steps, list)):
        print(f"Loaded object is a Pipeline with {len(model.steps)} steps.")
        # Assume the Decision Tree (or final estimator) is the last step
        estimator = model.steps[-1][1]
        
    print(f"Model Type: {type(estimator).__name__}")
    
    # Check if the extracted object is a Decision Tree (or related structure)
    if hasattr(estimator, 'tree_') and hasattr(estimator, 'feature_importances_'):
        
        # 1. Structural Information
        n_features = estimator.n_features_in_ if hasattr(estimator, 'n_features_in_') else len(estimator.feature_importances_)
        
        print(f"Features Model was Trained On: {n_features}")
        print(f"Max Depth of Tree: {estimator.tree_.max_depth}")
        print(f"Number of Leaf Nodes: {estimator.tree_.n_leaves}")

        # 2. Feature Importances (Shows which features the model *actually* uses)
        importances = estimator.feature_importances_
        
        # Create a DataFrame to easily sort and display non-zero importances
        feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Filter for features that actually contributed to the splits (Importance > 0)
        used_features = importance_df[importance_df['Importance'] > 0].sort_values(
            by='Importance', ascending=False
        )

        num_used = len(used_features)
        print(f"\nTotal Features with Non-Zero Importance (Actually Used in Splits): {num_used}")
        
        if num_used > 0:
            print("\nTop 5 Feature Importances (if used):")
            print(used_features.head(5).to_string(index=False))
        else:
            print("\nThis model uses no features (e.g., it might be a simple constant predictor).")

    else:
        print(f"WARNING: The final step ({type(estimator).__name__}) does not appear to be a standard Scikit-learn tree model.")

if __name__ == "__main__":
    
    # Run the inspection for both models
    inspect_model(POLICY_MODEL_PATH, "Policy")
    inspect_model(BASELINE_MODEL_PATH, "Baseline")

    print("\n\n--- INSPECTION COMPLETE ---")
    print("If both models show the same Max Depth and only 1 non-zero feature,")
    print("that explains why their file sizes (0.002 MB) are identical.")
