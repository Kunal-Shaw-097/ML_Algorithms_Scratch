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
        self.exp_x = None
        self.sum_exp_x = None

    def forward(self, x : np.array) -> np.array:
        """
        e^x / sum(e^i for i in range(0, num_classes))
        """
        self.exp_x = np.exp(x)
        self.sum_exp_x = np.sum(x, axis = 1, keepdims=True)
        self.probs = x / self.sum_exp_x
        return self.probs

    def backward(self) -> np.array:
        """
        (e^x * (sum_exp_x) - (exp^x)^2 ) /  (sum_exp_x)^2
        """
        grads = (self.exp_x * (self.sum_exp_x) - self.exp_x**2 )/ self.sum_exp_x**2 
        return grads
    

class CrossEntropy():
    """
    Cross Entropy
    """
    def __init__(self)-> None:
        self.x = None
        self.mask = None

    def forward(self, x : np.array, y : np.array) -> np.array:
        self.x = x
        self.y = y
        # Clip x to prevent log(0) which gives -inf
        x_clipped = np.clip(x, 1e-12, 1 - 1e-12)
        # Compute the loss
        batch_size = x.shape[0]
        correct_logprobs = -np.log(x_clipped[range(batch_size), y])
        loss = np.sum(correct_logprobs) / batch_size
        return loss
    
    def backward(self) -> np.array:
        batch_size = self.x.shape[0]
        grads = np.zeros_like(self.x)
        grads[range(batch_size), self.y] = 1/self.x[range(batch_size), self.y]
        return grads

class Linear():
    """
    Linear Layer
    """
    def __init__(self, in_features : int, out_features : int, bias : bool = False) -> None:
        self.weight = np.random.randn(in_features, out_features)
        self.bias = None
        self.x = None
        if bias:
            self.bias = np.zeros((1 , out_features))

    def forward(self, x : np.array) -> np.array:
        self.x = x
        x = np.dot(x , self.weight) + self.bias
        return x
    
    def backward(self):
        return self.x


class MLP():
    """
    A simple implementation of a Multi Layer Perceptron (only one hidden layer)
    """
    def __init__(self, in_features : int = 10, hidden_dim : int = 100, num_classes : int = 2) -> None:
        self.W1 = Linear(in_features, hidden_dim)
        self.W2 = Linear(hidden_dim, num_classes)
        self.act = ReLU()
        self.softmax = Softmax()
        self.loss = CrossEntropy()

    def forward(self, x : np.array) -> np.array:
        x = self.W1.forward(x) 
        x = self.act.forward(x)
        x = self.W2.forward(x)
        x = self.softmax.forward(x)
        return x
    
    def backward(self):

        pass

    def train(self, X: np.array, Y : np.array, epochs: int = 10, verbose : bool = True) -> None:
        for i in range(epochs):
            preds = self.forward(X)
            loss = self.loss.forward(Y, preds)
            if verbose :
                if i % 10 == 9 :
                    print(f"Loss at epoch {i + 1} is: {loss}")
            self.backwards(X, Y, preds)

