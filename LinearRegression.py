import numpy as np
from typing import Tuple

class LinearRegression:
    def __init__(self, learning_rate: float, iterations: int):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.W = None
        self.b = None
        self.X = None
        self.Y = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'LinearRegression':
        self.m, self.n = X.shape
        self.W = np.zeros((self.n, 1))
        self.b = 0
        self.X = X
        self.Y = Y

        for _ in range(self.iterations):
            self.update_weights()
        return self

    def update_weights(self) -> None:
        Y_pred = self.predict(self.X)
        dW = -2 * self.X.T.dot(self.Y - Y_pred) / self.m
        db = -2 * np.sum(self.Y - Y_pred) / self.m
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.W) + self.b

    def get_parameters(self) -> Tuple[np.ndarray, float]:
        return self.W, self.b

    def mean_squared_error(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        return np.mean((Y_true - Y_pred) ** 2)
    
