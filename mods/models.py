import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

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

class ForestPredictor:
    name = 'rf'
    def __init__(self):
        self.model = RandomForestRegressor(
            max_depth=10,
            max_samples=0.1,
            n_jobs=-1,
            random_state=143
        )
    def fit(self, X, y):
        self.last = X.iloc[-1]
        self.model.fit(X, y)
        return self
    def predict(self):
        return self.model.predict([self.last])[0]

class LSTMPredictor:
    name = 'nn'
    def fit(self, X, y):
        self.last = X.iloc[-1].values.reshape(1,1,X.shape[1])
        dim = min(int(X.shape[0]/4),1)
        self.mdl = tf.keras.models.Sequential([
            tf.keras.Input(X.shape),
            tf.keras.layers.LSTM(X.shape[1], return_sequences=True),
            tf.keras.layers.LSTM(X.shape[1], return_sequences=True),
            tf.keras.layers.LSTM(dim),
            tf.keras.layers.Dense(y.shape[1])
        ])
        self.mdl.compile('adam', 'mse', ['mse'])
        X = np.array(X, dtype=np.float32).reshape(X.shape[0],1,X.shape[1])
        y = np.array(y, dtype=np.float32)
        self.mdl.fit(X, y, epochs=20, verbose=0)
        return self
    def predict(self):
        return self.mdl.predict(self.last, verbose=0)[0]
