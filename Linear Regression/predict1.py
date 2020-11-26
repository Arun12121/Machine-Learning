import numpy as np
import csv
import sys
from sklearn.metrics import mean_squared_error

"""
Predicts the target values for data in the file at 'test_X_file_path', using the weights learned during training
Writes the predicted values to the file named "predicted_test_Y.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a proper implementation. Modify it based on the requirements of the project.
"""
def predict(test_X_file_path):

    # Load Test Data
    Xi = np.genfromtxt(test_X_file_path, delimiter=',', skip_header=1)
    test_X = np.ones((len(Xi), 5))
    test_X[:, 1:] = Xi
    W = np.array([-153.91168068, -22.98896794, 60.9467379, 31.81349104, 3.79769093])
    
    # Predict Target Variables
    """
    You can make use of any other helper functions which might be needed.
    Make sure all such functions are submitted in SubmissionCode.zip and imported properly.
    """

    # Write Outputs to 'predicted_test_Y.csv' file

    predicted_Y_values = np.matmul(test_X, W)
    with open('predicted_test_Y.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for y in predicted_Y_values:
            writer.writerow([y])

    Y = np.genfromtxt("train_Y_lr.csv", delimiter=',')
    print(mean_squared_error(Y, predicted_Y_values))

if __name__ == "__main__":
    predict(sys.argv[1])