import numpy as np
import pickle
import math
from validate import validate
from sklearn.metrics import accuracy_score


class Node:
    def __init__(self, predicted_class, depth):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.depth = depth
        self.left = None
        self.right = None

def scan_X(M):
    X = []
    for i in range(M):
        X.append([float(i) for i in input().split()])
    return X

def calculate_gini_index(Y_subsets):
    gini_index = 0
    total_instances = sum(len(Y) for Y in Y_subsets)
    for Y in Y_subsets:
        m = len(Y)
        if m == 0:
            continue
        count = [Y.count(c) for c in classes]
        gini = 1.0 - sum((n / m) ** 2 for n in count)
        gini_index += (m / total_instances)*gini
    
    return gini_index


def split_data_set(X, Y, feature_index, value):
    left_X = []
    right_X = []
    left_Y = []
    right_Y = []
    for i in range(len(X)):
        if X[i][feature_index] < value:
            left_X.append(X[i])
            left_Y.append(Y[i])
        else:
            right_X.append(X[i])
            right_Y.append(Y[i])
    
    return left_X, left_Y, right_X, right_Y


def get_best_split(X, Y):
    X = np.array(X)
    best_gini_index = 9999
    best_feature_index = None
    best_threshold = None
    if len(X) > 0:
        for i in range(len(X[0])):
            thresholds = set(sorted(X[:, i]))
            for t in thresholds:
                left_X, left_Y, right_X, right_Y = split_data_set(X, Y, i, t)
                if len(left_X) == 0 and len(right_X) == 0:
                    continue
                gini_index = calculate_gini_index([left_Y, right_Y])
                if gini_index < best_gini_index:
                    best_gini_index, best_feature_index, best_threshold = gini_index, i, t
    return best_feature_index, best_threshold


def construct_tree(X, Y, depth = 0):
    predicted_class = classes[np.argmax([np.sum(Y == c) for c in classes])]
    node = Node(predicted_class, depth)
    
    if len(set(Y)) == 1:
        return node
    if depth >= max_depth:
        return node
    if len(Y) <= min_size:
        return node

    feature_index, threshold = get_best_split(X, Y)
    if feature_index is None or threshold is None:
        return node

    node.feature_index = feature_index
    node.threshold = threshold
    
    left_X, left_Y, right_X, right_Y = split_data_set(X, Y, feature_index, threshold)

    node.left = construct_tree(np.array(left_X), np.array(left_Y))
    node.right = construct_tree(np.array(right_X), np.array(right_Y))
    return node

def print_tree(node):
    if node.left is not None and node.right is not None:
        print("X" + str(node.feature_index) + " " + str(node.threshold))
    if node.left is not None:
        print_tree(node.left)
    if node.right is not None:
        print_tree(node.right)


def predict(root, X):
    node = root
    while node.left:
        if X[node.feature_index] < node.threshold:
            node = node.left
        else:
            node = node.right
    return node.predicted_class

def predict_batch(root, test_X):
    pred_Y = []
    for x in test_X:
        pred_Y.append(predict(root, x))
    return pred_Y


if __name__ == "__main__":
    X = np.genfromtxt("train_X_de.csv", delimiter=',', dtype=np.float64, skip_header=1)
    Y = np.genfromtxt("train_Y_de.csv", delimiter=',')

    max_depth = 5
    min_size = 3

    ntrain = int(math.floor((100 - 5) / 100.0 * len(X)))
    train_X = X
    train_Y = Y
    test_X = X
    test_Y = Y
    
    classes = list(set(Y))

    root = construct_tree(train_X, train_Y)
    #pickle.dump(root, open("MODEL_FILE.sav", 'wb'))
    model = pickle.load(open("MODEL_FILE.sav", 'rb'))

    pred_Y = predict_batch(model, test_X)
    accumat = accuracy_score(test_Y, pred_Y)
    print(accumat)