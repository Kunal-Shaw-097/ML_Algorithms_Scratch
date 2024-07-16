import numpy as np


class ReLU():
    """
    ReLU activation implemetation
    """
    def __init__(self) -> None:
        self.mask = None
    
    def forward(self, x : np.array) -> np.array:
        self.mask = x > 0
        x[~self.mask] = 0
        return x
    
    def backward(self) -> np.array:
        return self.mask.astype(np.float32)


class Softmax():
    """
    Softmax activation implementation
    """
    def __init__(self) -> None:
        self.logits = None

    def forward(self, x : np.array) -> np.array:
        x = np.exp(x)
        self.logits = x / np.sum(x, axis = 1, keepdims=True)
        return self.logits

    def backward(self) -> np.array:
        pass

class MLP():
    """
    A simple implementation of a Multi Layer Perceptron (only one hidden layer)
    """
    def __init__(self, in_features : int = 10, hidden_dim : int = 100, num_classes : int = 2):
        self.W1 = np.random.randn((in_features, hidden_dim))
        self.W2 = np.random.randn((hidden_dim, num_classes))
        self.act = ReLU()
