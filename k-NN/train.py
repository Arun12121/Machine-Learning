import numpy as np
import csv
import math
from sklearn.metrics import f1_score

def compute_ln_norm_distance(vector1, vector2, n):
    """
    Arguments:
    vector1 -- A 1-dimensional array of size > 0.
    vector2 -- A 1-dimensional array of size equal to size of vector1
    n       -- n in Ln norm distance (>0)
    """
    vector_len = len(vector1)
    diff_vector = []

    for i in range(0, vector_len):
      abs_diff = abs(vector1[i] - vector2[i])
      diff_vector.append(abs_diff ** n)
    ln_norm_distance = (sum(diff_vector))**(1.0/n)
    return ln_norm_distance


def find_k_nearest_neighbors(train_X, test_example, k, n_in_ln_norm_distance):
    """
    Returns indices of 1st k - nearest neighbors in train_X, in order with nearest first.
    """
    indices_dist_pairs = []
    index= 0
    for train_elem_x in train_X:
      distance = compute_ln_norm_distance(train_elem_x, test_example,n_in_ln_norm_distance)
      indices_dist_pairs.append([index, distance])
      index += 1
    indices_dist_pairs.sort(key = lambda x: x[1])
    top_k_pairs = indices_dist_pairs[0:k]
    top_k_indices = [i[0] for i in top_k_pairs]
    return top_k_indices


def classify_points_using_knn(train_X, train_Y, test_X, n_in_ln_norm_distance, k):
    test_Y = []
    for test_elem_x in test_X:
      top_k_nn_indices = find_k_nearest_neighbors(train_X, test_elem_x, k,n_in_ln_norm_distance)
      top_knn_labels = []
      for i in top_k_nn_indices:
        top_knn_labels.append(train_Y[i])
      most_frequent_label = max(set(top_knn_labels), key = top_knn_labels.count)
      test_Y.append(most_frequent_label)
    return test_Y

def calculate_accuracy(predicted_Y, actual_Y):
  '''
    total_num_of_observations = len(predicted_Y)
    num_of_values_matched = 0
    for i in range(0,total_num_of_observations):
        if(predicted_Y[i] == actual_Y[i]):
            num_of_values_matched +=1
    return float(num_of_values_matched)/total_num_of_observations'''
  return f1_score(actual_Y, predicted_Y, average='weighted')


def get_best_k_using_validation_set(train_X, train_Y, validation_split_percent,n_in_ln_norm_distance):
    """
    Returns best value of k which gives best accuracy
    """
    nofobs = int(math.floor((100 - validation_split_percent) / 100.0 * len(train_X)))
    X = train_X[:nofobs]
    Y = train_Y[:nofobs]
    Xv = train_X[nofobs:]
    Yv = train_Y[nofobs:]
    
    accuracy = []
    for k in range(1, nofobs+1):   
        pred_y = classify_points_using_knn(X, Y, Xv, n_in_ln_norm_distance, k)
        accuracy.append(calculate_accuracy(pred_y, Yv))
    print(max(accuracy))
    return (accuracy.index(max(accuracy)) + 1)


if __name__ == "__main__":

    X = np.genfromtxt("train_X.csv", delimiter=',', skip_header=1)
    train_X = np.ones((160, 8))
    train_X[:, 1:] = X

    train_Y = np.genfromtxt("train_Y.csv", delimiter=',')

    validation_split_percent = 20
    n_ln_norm = 1

    best_k = get_best_k_using_validation_set(train_X, train_Y,validation_split_percent ,n_ln_norm)
    print(best_k)