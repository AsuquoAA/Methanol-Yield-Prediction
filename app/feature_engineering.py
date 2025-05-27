from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


# Custom transformer to engineer new features from existing columns
class EngineerFeatures(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        # No fitting necessary; stateless transformer
        return self

    def transform(self,X):
        X=X.copy()
        # Create new feature: pressure-to-temperature ratio
        X['Pressure/Temp'] = X['Pressure (bar)']/X['Temperature (K)']
        # Create new feature: total residence time
        X['Total res time'] = X['Residence Time (s)_1']+X['Residence Time (s)_2']
        # Create new feature: pressure-to-total-time ratio
        X['Press/Total time'] = X['Pressure (bar)']/X['Total res time']
        return X    

# Custom transformer to drop redundant or less useful features
class DropRedundantFeatures(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        # No fitting necessary; stateless transformer
        return self

    def transform(self,X):
        X=X.copy()
        # Drop features that may be redundant or replaced by engineered features
        X = X.drop(['Residence Time (s)_2','Pressure (bar)','Total res time'],axis='columns')
        return X   