import numpy as np
import math

class base():
    def mse(self, Y, preds):
        return np.mean(0.5 * ((Y - preds)**2))

    def rmse(self, Y, preds):
        return np.sqrt(np.mean(((Y - preds)**2)))
    
    def bce(self, Y, preds):
        return - np.mean(Y*np.log(preds) + (1- Y)*np.log(1 - preds))

    def tanh(self, logits):
        e = np.exp(2 * logits)
        preds = (e - 1)/(e + 1)
        return preds
    
    def sigmoid(self, logits):
        preds = 1/ (1 + np.exp(-logits))
        return preds
    
    def binary_cross_entropy(self, Y, preds):
        return np.mean(Y * np.log(Y) + (1 - Y) * np.log(1 - Y))


class LinearRegressor(base):
    def __init__(self, learning_rate : float = 0.01, loss : str = 'mse'):
        self.lr = learning_rate
        self.loss_name = loss
        if loss == 'rmse' :
            self.loss = self.rmse
        elif loss == 'mse':
            self.loss = self.mse
        else :
            raise Exception(f"Could not find loss {loss}")

    def backwards(self, X, Y , preds):
        if self.loss_name == 'mse':
            gradients_weights =  -np.matmul(X.T, (Y - preds)) / len(Y)
            self.weights -= gradients_weights * self.lr
        elif self.loss_name == 'rmse':
            error = (preds - Y)
            mse_loss = np.mean(error ** 2)
            rmse_loss = np.sqrt(mse_loss)
            gradients_weights = np.matmul(X.T, error / rmse_loss) / len(Y)
            self.weights -= gradients_weights * self.lr
     
    def fit(self, X , Y, epochs : int = 100, verbose : bool = True):
        num_features = X.shape[1]
        self.weights = np.random.uniform(-6/(num_features + 1), 6/(num_features + 1), (num_features, 1))
        self.bias = np.zeros((1,1))
        for i in range(epochs):
            preds = np.matmul(X , self.weights) + self.bias
            loss = self.loss(Y, preds)
            if verbose :
                if i % 10 == 9 :
                    print(f"Loss at epoch {i + 1} is: {loss}")
            self.backwards(X, Y, preds)
    
    def predict(self, X):
        preds = np.matmul(X, self.weights)
        #preds = np.where(preds > threshold, 1, 0)
        preds = preds[:, 0]
        return preds
    

class LogisticRegression(base):
    def __init__(self, learning_rate : float = 0.01, loss : str = 'bce', activation: str = 'sigmoid'):
        self.lr = learning_rate
        self.loss_name = loss
        self.activation_name = activation

        if loss == 'bce':
            self.loss = self.bce
        elif loss == 'ce':
            raise NotImplementedError
        else :
            raise Exception(f"Could not find loss {loss}")
        
        if activation == 'sigmoid':
            self.act = self.sigmoid
        else :
            raise Exception(f"Could not find activation {activation}")

    def backwards(self, X, Y , preds):
        if self.loss_name == 'bce':
            gradients_weights =  -np.matmul(X.T, (Y - preds))/ len(Y)                          ## (Y - preds)/(Y*(1 - Y)) * (Y*(1-Y)) = Y - preds       
            self.weights -= gradients_weights * self.lr

    def fit(self, X , Y, epochs : int = 100, verbose : bool = True):
        num_features = X.shape[1]
        self.weights = np.random.uniform(-6/(num_features + 1), 6/(num_features + 1), (num_features, 1))
        self.bias = np.zeros((1,1))
        for i in range(epochs):
            logits = np.matmul(X , self.weights) + self.bias
            preds = self.act(logits)
            loss = self.loss(Y, preds)
            if verbose :
                if i % 10 == 9 :
                    print(f"Loss at epoch {i + 1} is: {loss}")
            self.backwards(X, Y, preds)
    
    def predict(self, X, threshold : float = 0.5):
        preds = np.matmul(X, self.weights)
        preds = np.where(preds > threshold, 1, 0)
        preds = preds[:, 0]
        return preds
    

