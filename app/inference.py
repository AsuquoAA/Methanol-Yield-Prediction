import pandas as pd
import joblib
from app.target_transformation import load_boxcox_transformer


# Load model pipeline and boxcox transformer
pipeline = joblib.load('model.joblib')
boxcox = load_boxcox_transformer()

data = {
    'Pressure (bar)': [66, 105, 20, 105],          # example values
    'Temperature (K)': [566, 560, 490, 330],
    'Residence Time (s)_1': [20, 25, 22, 23],
    'Residence Time (s)_2': [5, 14, 7, 8]
}

# New input
X_new = pd.DataFrame(data)

# Predict
y_pred_transformed = pipeline.predict(X_new)
y_pred = boxcox.inverse_transform(y_pred_transformed.reshape(-1, 1))

print(y_pred.flatten())