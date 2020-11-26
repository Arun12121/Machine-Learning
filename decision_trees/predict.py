import numpy as np
import csv
import sys
import arff
import pickle

from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training
Writes the predicted values to the file named "predicted_test_Y_dt.arff". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a proper implementation. Modify it based on the requirements of the project.
"""

def import_data_and_model(test_X_file_path, model_file_path):
    test_X_arff=arff.load(open(test_X_file_path,'r'))
    test_X=test_X_arff['data']
    test_X=np.asarray(test_X)
    model = pickle.load(open(model_file_path, 'rb'))

    return test_X, model

def predict_target_values(test_X, model):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.


def write_to_arff_file(pred_Y, predicted_Y_file_name):
    arff_data={
        'data':predicted_Y_values, 
        'relation':test_X_arff['relation'], 'description':'', 
        'attributes':[('class',['True','False'])]
        }
    with open('./predicted_test_Y_dt.arff','w') as arff_file:
        arff.dump(arff_data,arff_file)


def predict(test_X_file_path):
    test_X, model = import_data_and_model(test_X_file_path, 'MODEL_FILE.sav')
    pred_Y = predict_target_values(test_X, model)
    write_to_arff_file(pred_Y,  "predicted_test_Y_dt.arff")
    


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_dt.csv") 