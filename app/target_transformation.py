import joblib


def load_boxcox_transformer(path='./objects/yield_boxcox_transformer.pkl'):
    return joblib.load(path)    