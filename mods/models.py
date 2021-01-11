import numpy as np
from sklearn.linear_model import LinearRegression

class AlwaysZeroPredictor:
    name = 'zz'
    def fit(self, X, y):
        self.outshape = 1 if len(y.shape) == 1 else y.shape[1]
        return self
    def predict(self):
        return [0] * self.outshape

class SimpleLinearPredictor:
    name = 'lm'
    def __init__(self):
        self.model = LinearRegression(fit_intercept=False, n_jobs=-1)
    def fit(self, X, y):
        self.last = X.iloc[-1]
        self.model.fit(X, y)
        return self
    def predict(self):
        return self.model.predict([self.last])[0]
