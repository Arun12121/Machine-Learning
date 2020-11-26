from os import path
import numpy as np
import csv
import train

def preprocessing(s):
    s_with_no_spl_chars = ""
    for i in range(len(s)): 
        if (ord(s[i]) >= ord('A') and
            ord(s[i]) <= ord('Z') or 
            ord(s[i]) >= ord('a') and 
            ord(s[i]) <= ord('z') or
            ord(s[i]) == ord(' ')):
            s_with_no_spl_chars += s[i]
    s = s_with_no_spl_chars
    s = s.lower()
    s =' '.join(s.split())
    return s

def check_file_exits(predicted_test_Y_file_path):
    if not path.exists(predicted_test_Y_file_path):
        raise Exception("Couldn't find '" + predicted_test_Y_file_path +"' file")


def check_format(test_X_file_path, predicted_test_Y_file_path):
    pred_Y = []
    with open(predicted_test_Y_file_path, 'r') as file:
        reader = csv.reader(file)
        pred_Y = list(reader)
    pred_Y = np.array(pred_Y)

    test_X = []
    for line in open(test_X_file_path):
        test_X.append(train.preprocessing(line))

    if pred_Y.shape != (len(test_X), 1):
        raise Exception("Output format is not proper")


def check_accuracy(actual_test_Y_file_path, predicted_test_Y_file_path):
    pred_Y = np.genfromtxt(predicted_test_Y_file_path, delimiter=',')
    actual_Y = np.genfromtxt(actual_test_Y_file_path, delimiter=',', dtype=np.int)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(actual_Y, pred_Y)
    print("Accuracy", accuracy)
    return accuracy


def validate(test_X_file_path, actual_test_Y_file_path):
    predicted_test_Y_file_path = "predicted_test_Y_nb.csv"
    
    check_file_exits(predicted_test_Y_file_path)
    check_format(test_X_file_path, predicted_test_Y_file_path)
    check_accuracy(actual_test_Y_file_path, predicted_test_Y_file_path)