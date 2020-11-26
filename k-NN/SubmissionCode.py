import numpy as np
import csv
import sys
import math
"""
Predicts the target values for data in the file at 'test_X_file_path'.
Writes the predicted values to the file named "predicted_test_Y.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a proper implementation. Modify it based on the requirements of the project.
"""
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

def predict(test_X_file_path):

    # Load Test Data
    X = np.genfromtxt(test_X_file_path, delimiter=',', skip_header=1)
    test_X = np.ones((X.shape[0], 8))
    test_X[:, 1:] = X

    X = np.genfromtxt("train_X.csv", delimiter=',', skip_header=1)
    train_X = np.ones((160, 8))
    train_X[:, 1:] = X

    train_Y = np.genfromtxt("train_Y.csv", delimiter=',')
    # Load Model Parameters
    k = 2
    n = 1
    
    # Predict Target Variables
    """
    You can make use of any other helper functions which might be needed.
    Make sure all such functions are submitted in SubmissionCode.zip and imported properly.
    """

    # Write Outputs to 'predicted_test_Y.csv' file

    predicted_Y_values = classify_points_using_knn(train_X, train_Y, test_X, n, k)
    with open('predicted_test_Y.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for y in predicted_Y_values:
            writer.writerow([y])


if __name__ == "__main__":
    predict(sys.argv[1])