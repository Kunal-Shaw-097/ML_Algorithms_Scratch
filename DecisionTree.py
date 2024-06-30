import numpy as np

class Node():
    def __init__(self, feature = None, threshold = None, left = None, right = None, value= None):
        self.feature = feature                # feature name
        self.threshold = threshold            # value at which to threshold    
        self.value = value                    # if leaf node what is the prediction ?
        self.left = left                      # left child node
        self.right = right                    # right child node. Duh

class DecisionTree():
    def __init__(self, max_depth = 5, min_samples = 5):
        self.root = None
        self.max_depth = max_depth
        self.min_samples = min_samples

    def get_entropy(self, Y):    
        unique_values = np.unique(Y)
        entropy = 0
        for value in unique_values :
            labels = Y[Y == value]
            p = len(labels)/len(Y)
            entropy += -(p * np.log(p))
        return entropy

    def get_gain(self, Y, left_Y , right_Y):

        information_gain = 0

        parent_entropy = self.get_entropy(Y)
        left_entropy = self.get_entropy(left_Y)
        right_entropy = self.get_entropy(right_Y)

        left_weight = len(left_Y)/len(Y)
        right_weight = len(right_Y)/len(Y)

        weighted_entropy = left_entropy * left_weight  + right_entropy * right_weight

        information_gain = parent_entropy - weighted_entropy
        return information_gain


    def split_data(self, data, threshold, i):
        left_data = []
        right_data = []
        for row in data:
            if row[i] < threshold :
                left_data.append(row) 
            else :
                right_data.append(row) 
        return np.array(left_data), np.array(right_data)

    def find_best_split(self, data):
        num_features = data.shape[1] - 1
        max_gain = -1
        best_feature = None
        best_threshold = None
        left_data = None
        right_data = None
        for i in range(num_features):
            unique_values = np.unique(data[:,i])
            for threshold in unique_values:
                temp_left_data, temp_right_data = self.split_data(data, threshold, i)
                if len(temp_left_data) and len(temp_right_data):
                    gain = self.get_gain(data[:, -1], temp_left_data[:, -1], temp_right_data[:, -1])     
                    if gain > max_gain :
                        best_feature = i
                        best_threshold = threshold
                        max_gain = gain
                        left_data = temp_left_data
                        right_data = temp_right_data
                        #print('best partitioning into:',len(left_data), len(right_data))              
        return left_data, right_data, best_feature, best_threshold, max_gain
    
    def calculate_value(self, Y):
        Y = list(Y)
        value = max(Y, key= Y.count)
        return value

    def build_tree(self, data, current_depth = 0):
        num_samples = data.shape[0]
        if current_depth <= self.max_depth and num_samples >= self.min_samples:
            # split data
            left_dataset, right_dataset, feature, threshold, gain = self.find_best_split(data)
            #print(f"at depth {current_depth}: ", len(left_dataset), len(right_dataset))
            if gain > 0:
                left_node = self.build_tree(left_dataset, current_depth= current_depth+1)
                right_node = self.build_tree(right_dataset, current_depth= current_depth+1)
                return Node(feature, threshold, left_node, right_node)
        leaf_value = self.calculate_value(data[:, -1])
        return Node(value=leaf_value)


    def fit(self, X, Y):
        data = np.concatenate([X, Y], axis = 1)
        self.root = self.build_tree(data)

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.make_predictions(x, self.root))
        return np.array(predictions)

    def make_predictions(self, x, node):
        if node.value is not None:
            return node.value
        else :
            feature = node.feature
            if x[feature] < node.threshold:
                return self.make_predictions(x, node.left)
            else:
                return self.make_predictions(x, node.right)