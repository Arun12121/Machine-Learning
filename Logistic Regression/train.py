import numpy as np

def scan_X_vector(M):
    X = []
    for i in range(0, M):
        Xi = [float(j) for j in input().split()]
        X.append(Xi)
    X = np.array(X)
    return X


def scan_Y_vector():
    Y = [float(i) for i in input().split()]
    Y = np.array(Y)
    return Y

def scan_W_vector():
    W = [[float(i)] for i in input().split()]
    W = np.array(W)
    return W

def sigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    return s

def compute_cost(X, Y, W, b):
    """
    Arguments:
    X -- data of shape (number of observations ,number of features)
    Y --  true "label" values (containing 1 for positive class, 0 for negative class) of shape (number of observations, 1)
    W -- weights for features of shape (number of features, 1)

    Return:
    cost value for logistic regression
    """  
    m = X.shape[0]
    A = sigmoid(np.matmul(X, W) + b)
    cost = -1.0/m * np.sum(np.matmul(Y.T , np.log(A)) + np.matmul((1-Y).T, np.log(1-A)))
    return cost

def compute_gradient_of_cost_function(X, Y, W, b):
    """
    Arguments:
    X -- data of shape (number of observations ,number of features)
    Y -- true "label" values (containing 1 for positive class, 0 for negative class) of shape (number of observations, 1)
    W -- weights for features of shape (number of features, 1)

    Return:
    dW -- gradient of the cost function with respect to W, with same shape as W
    """
    m = X.shape[0]
    A = sigmoid(np.matmul(X, W) + b)
    print(sigmoid(np.matmul(X, W)))
    dw = np.matmul(X.T, (A-Y)) / m * 1.0
    db = np.sum(A-Y) / m * 1.0
    return [dw, db]

def train(X, Y):

    W = np.zeros(X.shape[1])
    b = 0
    learning_rate = 0.0001

    for i in range(100):
        dw, db = compute_gradient_of_cost_function(X, Y_true, W, b)
        db = np.round(db, 3)
        W -= learning_rate * dw
        b -= learning_rate * db
        #print(compute_cost(X, Y, W, b))
    

if __name__ == "__main__":
    X = np.genfromtxt("train_X.csv", delimiter=',', skip_header=1)
    Y_true = np.genfromtxt("train_Y.csv", delimiter=',')
    Y = np.array([np.where(Y_true != 0, -1, Y_true) + 1, np.where(Y_true != 1, 0, Y_true), np.where(Y_true != 2, 1, Y_true) - 1, np.where(Y_true != 3, 2, Y_true) - 2])
    train(X, Y[0])