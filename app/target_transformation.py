import joblib


# Function to load the saved Box-Cox transformer
# This transformer is used to inverse-transform predicted values (e.g., converting them back from Box-Cox space)
def load_boxcox_transformer(path='./objects/yield_boxcox_transformer.pkl'):
    return joblib.load(path)    