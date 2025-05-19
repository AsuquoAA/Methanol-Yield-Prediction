from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class EngineerFeatures(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X=X.copy()
        X['Pressure/Temp'] = X['Pressure (bar)']/X['Temperature (K)']
        X['Total res time'] = X['Residence Time (s)_1']+X['Residence Time (s)_2']
        X['Press/Total time'] = X['Pressure (bar)']/X['Total res time']
        return X    

class DropRedundantFeatures(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X=X.copy()
        X = X.drop(['Residence Time (s)_2','Pressure (bar)','Total res time'],axis='columns')
        return X   