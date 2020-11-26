from os import path
import numpy as np
import csv
import arff


def check_file_exits(predicted_test_Y_file_path):
    if not path.exists(predicted_test_Y_file_path):
        raise Exception("Couldn't find '" + predicted_test_Y_file_path +"' file")


def check_format(test_X_file_path, predicted_test_Y_file_path):
    pred_Y = np.array(arff.load(open(predicted_test_Y_file_path, newline=''), 'rb')['data'])

    print(pred_Y.shape)

    test_X_arff=arff.load(open(test_X_file_path,'r'))
    test_X=test_X_arff['data']
    test_X=np.asarray(test_X)

    print(test_X.shape)

    if pred_Y.shape != (len(test_X), 1):
        raise Exception("Output format is not proper")


def check_accuracy(actual_test_Y_file_path, predicted_test_Y_file_path):
    pred_Y = np.array(arff.load(open(predicted_test_Y_file_path, newline=''), 'rb')['data'])
    actual_Y = np.array(arff.load(open(actual_test_Y_file_path, newline=''), 'rb')['data'])
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(actual_Y, pred_Y)
    print("Accuracy", accuracy)
    return accuracy


def validate(test_X_file_path, actual_test_Y_file_path):
    predicted_test_Y_file_path = "predicted_test_Y_dt.csv"
    
    check_file_exits(predicted_test_Y_file_path)
    check_format(test_X_file_path, predicted_test_Y_file_path)
    check_accuracy(actual_test_Y_file_path, predicted_test_Y_file_path)