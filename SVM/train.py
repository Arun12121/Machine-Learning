import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import math
import pickle
    

if __name__ == "__main__":
    train_X = np.genfromtxt("train_X_svm.csv", delimiter=',', skip_header=1)
    train_Y = np.genfromtxt("train_Y_svm.csv", delimiter=',')
    c = 13
    validation_split_percent = 20
    nofobs = int(math.floor((100 - validation_split_percent) / 100.0 * len(train_X)))
    clf = make_pipeline(StandardScaler(), SVC(C = c))
    clf.fit(train_X, train_Y)
    pickle.dump(clf, open('model.pkl', "wb"))