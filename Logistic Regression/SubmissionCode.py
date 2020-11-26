import numpy as np
import csv
import sys

"""
Predicts the target values for data in the file at 'test_X_file_path', using the weights learned during training
Writes the predicted values to the file named "predicted_test_Y.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a proper implementation. Modify it based on the requirements of the project.
"""
def predict(test_X_file_path):

    # Load Test Data
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', skip_header=1)

    # Load Model Parameters
    """
    You can load the weights/parameters by reading them from a csv file which is present in the SubmissionCode.zip
    """
    
    # Predict Target Variables
    """
    You can make use of any other helper functions which might be needed.
    Make sure all such functions are submitted in SubmissionCode.zip and imported properly.
    """

    # Write Outputs to 'predicted_test_Y.csv' file

    predicted_Y_values = np.array() #ToDo: Update this to contain predicted outputs
    with open('predicted_test_Y.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(predicted_Y_values)


if __name__ == "__main__":
    predict(sys.argv[1])