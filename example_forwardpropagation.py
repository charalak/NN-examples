import numpy as np
import matplotlib.pyplot as plt

Nclass = 1000

X1 = np.random.randn(Nclass, 2) + np.array([-1, -1])
X2 = np.random.randn(Nclass, 2) + np.array([0, 2])
X3 = np.random.randn(Nclass, 2) + np.array([-3, 2])

X = np.vstack([X1, X2, X3])
Y_true = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)


D = 2
M = 3
K = 3

# weights and bias term of hidden layer
W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
# weights and bias term of output layer
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)


def forward_propagation(X, W1, b1, W2, b2):
    # sigmoid on hidden layer's nodes: X - input ; Z - output
    Z = 1.0/(1+np.exp(-X.dot(W1)-b1))
    # output layer
    A = Z.dot(W2) + b2
    # perform softmax on output layer
    expA = np.exp(A)
    prob_multiclass = expA/expA.sum(axis=1, keepdims=True)
    return prob_multiclass


def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total+=1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct/n_total)


P_Y_X = forward_propagation(X, W1, b1, W2, b2)
print(P_Y_X)
P = np.argmax(P_Y_X, axis=1)
assert(len(P) == len(Y_true))

print(classification_rate(Y_true,P))