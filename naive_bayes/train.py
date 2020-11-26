import numpy as np
import pickle
import math

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

def compute_prior_probabilities(Y):
    ans = dict()
    for c in classes:
        ans[c] = Y.count(c) / len(Y)
    return ans

def get_class_wise_denominators_likelihood(X, Y):
    data = class_wise_words_frequency_dict(X, Y)
    ans = dict()
    l = set()
    for c in classes:
        l = l | set(data[c].keys())
    s = len(l)
    for c in classes:
        ans[c] = sum(data[c].values()) + s
    return ans

def class_wise_words_frequency_dict(X, Y):
    ans = dict()
    for i in range(len(Y)):
        if ans.get(Y[i]) == None:
            ans[Y[i]] = dict()
        for w in X[i].split():
            if ans[Y[i]].get(w) == None:
                ans[Y[i]][w] = 1
            else:
                ans[Y[i]][w] += 1
    return ans

def train(X, Y):
    return [class_wise_words_frequency_dict(X, Y), get_class_wise_denominators_likelihood(X, Y),  compute_prior_probabilities(Y), classes]

if __name__ == "__main__":
    X = []
    for line in open("train_X_nb.csv"):
        X.append(preprocessing(line))
    Y = list(np.genfromtxt("train_Y_nb.csv", delimiter=','))

    classes = list(set(Y))
    classes.sort()

    pickle.dump(train(X, Y), open("MODEL_FILE.sav", 'wb'))