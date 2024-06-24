from DecisionTree import DecisionTree
import numpy as np
from tqdm import tqdm


class RandomForest():
    def __init__(self, num_estimators : int = 2, max_depth : int = 3, min_samples : int = 10, boot_strap : float = None, feature_bagging : float = None):
        self.num_estimators = num_estimators
        self.boot_strap = 1 if boot_strap > 1 else boot_strap
        self.feature_bagging = 1 if feature_bagging > 1 else feature_bagging
        self.max_depth = max_depth 
        self.min_samples = min_samples
        self.features = []
        self.trees = []
    

    def fit(self, X , Y):
        rng_generator = np.random.default_rng()
        num_data, features_len = X.shape[0], X.shape[1]
        random_features_index = np.arange(features_len)
        random_data_points = np.arange(num_data)
        for i in tqdm(range(self.num_estimators), total = self.num_estimators):
            x = X
            if self.feature_bagging : 
                random_features_index = rng_generator.choice(features_len, int(self.feature_bagging * features_len), replace = False)
                x = x[:,random_features_index]
            if self.boot_strap : 
                random_data_points = rng_generator.choice(num_data, int(self.boot_strap * num_data), replace = True)
                x = x[random_data_points, :]
                y = Y[random_data_points, :]
            estimator = DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples)
            estimator.fit(x, y)
            self.trees.append(estimator)
            self.features.append(random_features_index)

    def get_max_occurance(self, predictons):
        max_occuring_preds = []
        for i in range(predictons.shape[1]):
            pred = predictons[:, i]
            counts = {}
            for j in pred :
                counts[j] = counts.get(j, 0) + 1
            max_occuring_preds.append(max(counts, key=counts.get))
        
        return np.array(max_occuring_preds)

    def predict(self, X) :
        if len(X.shape)  == 1 :
            X = X[None]
        if len(self.trees) == 0 :
            print("Please train the RandomForest Classifier first, by calling the 'fit()' method")
            return
        predictions = []
        for i,estimator in enumerate(self.trees):
            x = X
            if self.feature_bagging:
                features_index = self.features[i]
                x = x[:,features_index]
            predictions.append(estimator.predict(x))
        predictions = np.array(predictions)
        predictions = self.get_max_occurance(predictions)
        return predictions

        






