from sklearn.pipeline import Pipeline
from app.feature_engineering import EngineerFeatures,DropRedundantFeatures
import joblib


best_model = joblib.load('/Users/apple/Desktop/Methanol Synthesis Project/objects/best_model.pkl')

full_pipeline = Pipeline([
    ('engineer', EngineerFeatures()),
    ('drop', DropRedundantFeatures()),
    ('model', best_model.named_steps['model'])
])

joblib.dump(full_pipeline, 'model.joblib')