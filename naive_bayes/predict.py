import numpy as np
import csv
import sys
import pickle

from validate import validate
from train import preprocessing

"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_nb.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_model(test_X_file_path, model_file_path):
    X = []
    for line in open(test_X_file_path):
        X.append(preprocessing(line))
    model = pickle.load(open(model_file_path, 'rb'))
    return X, model

def compute_likelihood(test_X, c, class_wise_frequency_dict, class_wise_denominators):
    likelihood = 0
    words = test_X.split()
    for word in words:
        count = 0
        words_frequency = class_wise_frequency_dict[c]
        if word in words_frequency:
            count = class_wise_frequency_dict[c][word]
        likelihood += np.log((count + 1)/class_wise_denominators[c])
    return likelihood

def pred(test_X, classes, class_wise_frequency_dict, class_wise_denominators):
    ans = dict()
    for c in classes:
        ans[c] = compute_likelihood(test_X, c, class_wise_frequency_dict, class_wise_denominators)
    return max(ans, key = lambda x: ans[x])

def predict_target_values(test_X, model):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    class_wise_frequency_dict, class_wise_denominators, prior_probabilities, classes = model
    pred_Y = []
    for x in test_X:
        pred_Y.append(pred(x, classes, class_wise_frequency_dict, class_wise_denominators))
    return pred_Y


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = np.array(pred_Y)
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    # Load Model Parameters
    """
    Parameters of Naive Bayes include Laplace smoothing parameter, Prior probabilites of each class and values related to likelihood computation.
    """
    test_X, model = import_data_and_model(test_X_file_path, "MODEL_FILE.sav")
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_nb.csv")    


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_nb.csv") 