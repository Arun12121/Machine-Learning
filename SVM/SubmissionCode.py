import numpy as np
import csv
import sys
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import math
import joblib

"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training
Writes the predicted values to the file named "predicted_test_Y.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a proper implementation. Modify it based on the requirements of the project.
"""
def predict(test_X_file_path):

    # Load Test Data
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', skip_header=1)

    # Load Model Parameters
    model = joblib.load('model.pkl')
    predicted_Y_values = model.predict(test_X)
    with open('predicted_test_Y.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for y in predicted_Y_values:
            writer.writerow([y])


if __name__ == "__main__":
    predict(sys.argv[1])