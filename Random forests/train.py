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
    for _ in range(M):
        X.append([float(i) for i in input().split()])
    return X

def get_combined_train_XY(train_X, train_Y):
    train_XY = train_X.tolist()
    for i in range(len(train_X)):
        train_XY[i].append(train_Y[i])
    return train_XY

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

def err(model, xy):
    yp = predict(model, xy[:-1])
    if yp != xy[-1]:
        return 1
    else:
        return 0

def get_out_of_bag_error(models, train_XY, bootstrap_samples):
    M = P = len(train_XY)
    oob = 0.0
    for i in range(M):
        Z = 0
        s = 0
        for j in range(len(bootstrap_samples)):
            if train_XY[i] not in bootstrap_samples[j]:
                Z += 1
                s += err(models[j], train_XY[i])
        if Z != 0:
            oob += s / Z
        else:
            P -= 1
    return oob/P

def get_best_split(X, Y, feature_indices):
    X = np.array(X)
    best_gini_index = 9999
    best_feature_index = None
    best_threshold = None
    if len(X) > 0:
        for i in feature_indices:
            thresholds = set(sorted(X[:, i]))
            for t in thresholds:
                left_X, left_Y, right_X, right_Y = split_data_set(X, Y, i, t)
                if len(left_X) == 0 and len(right_X) == 0:
                    continue
                gini_index = calculate_gini_index([left_Y, right_Y])
                if gini_index < best_gini_index:
                    best_gini_index, best_feature_index, best_threshold = gini_index, i, t
    return best_feature_index, best_threshold

def get_split_in_random_forest(X, Y):
    num_features = 8
    feature_indices = []
    n = len(X[0])
    for _ in range(num_features):
        feature_indices.append(np.random.randint(0, n))
    return get_best_split(X, Y, feature_indices)

def construct_tree(X, Y, depth = 0):
    predicted_class = classes[np.argmax([np.sum(Y == c) for c in classes])]
    node = Node(predicted_class, depth)
    
    if len(set(Y)) == 1:
        return node
    if depth >= max_depth:
        return node
    if len(Y) <= min_size:
        return node

    feature_index, threshold = get_split_in_random_forest(X, Y)
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

def get_bootstrap_samples(train_XY, num_bootstrap_samples):
    samples = []
    for _ in range(num_bootstrap_samples):
        s = []
        for _ in range(len(train_XY)):
            s.append(train_XY[np.random.randint(0,len(train_XY))])
        samples.append(s)
    return samples


def predict(root, X):
    node = root
    while node.left:
        if X[node.feature_index] < node.threshold:
            node = node.left
        else:
            node = node.right
    return node.predicted_class

def get_trained_models_using_bagging(train_XY, bootstrap_samples):
    """
    Arguments:
    train_XY -- 2d list where each row contains X values of a train observation, along with Y value as last element
    num_bootstrap_samples -- number of bootstrap samples to generate and use

    Returns:
    list of trained models (of type BaseLearner) using bagging
    """
    model = []
    for sample in bootstrap_samples:
        train_X = [row[:-1] for row in sample]
        train_Y = [row[-1] for row in sample]
        root = construct_tree(train_X, train_Y)
        model.append(root)

    return model

def predict_using_bagging(models, test_X):
    pred_Y = []
    for model in models:
        pred_Y.append(predict(model, test_X))
    return max(set(pred_Y), key = pred_Y.count) 

def predict_batch(models, test_X):
    pred_Y = []
    for x in test_X:
        pred_Y.append(predict_using_bagging(models, x))
    return pred_Y


if __name__ == "__main__":
    X = np.genfromtxt("train_X_rf.csv", delimiter=',', dtype=np.float64, skip_header=1)
    Y = np.genfromtxt("train_Y_rf.csv", delimiter=',')

    max_depth = 100000000
    min_size = 1
    error_avg = np.zeros((11))
    m = 50
    ntrain = int(math.floor((100 - 20) / 100.0 * len(X)))
    train_X = X[:ntrain]
    train_Y = Y[:ntrain]
    test_X = X[ntrain:]
    test_Y = Y[ntrain:]
    train_XY = get_combined_train_XY(X, Y)
    classes = list(set(Y))
    m = 175

    bootstrap_samples = get_bootstrap_samples(train_XY, m)
    model = get_trained_models_using_bagging(train_XY, bootstrap_samples)

    pickle.dump(model, open("MODEL_FILE.sav", 'wb'))
    #model = pickle.load(open("MODEL_FILE.sav", 'rb'))

    #pred_Y = predict_batch(model, test_X)
    #print(m, accuracy_score(test_Y, pred_Y))