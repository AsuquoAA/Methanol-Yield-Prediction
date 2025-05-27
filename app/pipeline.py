from sklearn.pipeline import Pipeline
from app.feature_engineering import EngineerFeatures,DropRedundantFeatures
import joblib


# Load the best pre-tuned model previously saved
best_model = joblib.load('/Users/apple/Desktop/Methanol Synthesis Project/objects/best_model.pkl')

# Build the full pipeline combining feature engineering and the trained model
full_pipeline = Pipeline([
    # Step 1: Apply custom feature engineering transformations
    ('engineer', EngineerFeatures()),
    # Step 2: Drop redundant or unneeded features
    ('drop', DropRedundantFeatures()),
    # Step 3: Use the trained model (extracted from previously saved pipeline)
    ('model', best_model.named_steps['model'])
])

# Save the complete pipeline for inference use
joblib.dump(full_pipeline, 'model.joblib')