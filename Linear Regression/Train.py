import numpy as np
import csv
import sys
def compute_gradient_of_cost_function(X, Y, W):
    M = len(X)
    temp = np.matmul(X, W) - Y
    dW = (1.0/M) * np.matmul(np.transpose(X), temp)
    return dW

def cost_function(X, Y, W):
    m = len(X)
    temp = np.matmul(X, W) - Y
    ans = np.matmul(np.transpose(temp), temp)
    ans = (0.5 / m) * ans
    return ans


def train():

    # Load Test Data
    Xi = np.genfromtxt("train_X_lr.csv", delimiter=',', skip_header=1)
    X = np.ones((374, 5))
    X[:, 1:] = Xi
    Y = np.genfromtxt("train_Y_lr.csv", delimiter=',')

    learning_rate = 0.00022
    theta = np.matmul(X.T, X)
    theta = np.linalg.pinv(theta)
    theta = np.matmul(theta, np.matmul(X.T, Y))
    #W = np.array([-153.91168068, -22.98896794, 60.9467379, 31.81349104, 3.79769093])
    #for i in range(100):
    #    W -= learning_rate * compute_gradient_of_cost_function(X, Y, W)
    #    print(cost_function(X, Y, W))
    print(cost_function(X, Y, theta))
    print(theta)
    with open("Weights.csv", 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerow(theta)
        csv_file.close()

if __name__ == "__main__":
    train()