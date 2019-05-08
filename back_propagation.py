import numpy as np
import matplotlib.pyplot as plt


# calculation of probabilities of each class for each observation
def forward_propagation(X, W1, b1, W2, b2):
    # sigmoid on hidden layer's nodes: X - input ; Z - output
    Z = 1.0/(1+np.exp(-X.dot(W1)-b1))
    # output layer
    A = Z.dot(W2) + b2
    # perform softmax on the output layer
    expA = np.exp(A)
    prob_multiclass = expA/expA.sum(axis=1, keepdims=True)
    return prob_multiclass, Z


# average of correct prediction
def classification_rate(Y, P):
    cr = np.mean(Y == P)
    return cr


def derivative_w2(Z, T, Y):
    der_w2 = Z.T.dot(T-Y)
    return der_w2


def derivative_b2(T, Y):
    der_b2 = (T - Y).sum(axis=0)
    return der_b2


def derivative_w1(X, Z, T, Y, W2):
    Zprime = (T-Y).dot(W2.T) * Z * (1- Z)
    der_w1 = X.T.dot(Zprime)
    return der_w1


def derivative_b1(T, Y, W2, Z):
    der_b1 = ((T-Y).dot(W2.T) * Z * (1-Z)).sum(axis=0)
    return der_b1


def cost_logloss(T, Y):
    tot = T * np.log(Y)
    return tot.sum()


def main(Nclass=500, eta=10e-7, n_epochs=100000):
    # eta : learning rate
    D = 2
    M = 3
    K = 3

    X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
    X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
    X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])

    # target - ground true
    Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)
    N = len(Y)

    # one-hot-encoding of target
    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1

    # weights and bias term of hidden layer
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    # weights and bias term of output layer
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)



    costs = []
    for epoch in range(n_epochs):
        output, hidden = forward_propagation(X, W1, b1, W2, b2)
        # print every 100 epochs
        if epoch % 100 == 0:
            ll = cost_logloss(T, output)
            # prediction: 0, 1, 2 ... k
            P = np.argmax(output, axis=1)
            # classification rate
            r = classification_rate(Y, P)
            print("cost: {}, class_rate: {}".format(ll, r))
            costs.append(ll)

        W2 += eta * derivative_w2(hidden, T, output)
        b2 += eta * derivative_b2(T, output)

        W1 += eta * derivative_w1(X, hidden, T, output, W2)
        b1 += eta * derivative_b1(T, output, W2, hidden)

    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
    main(Nclass=500, eta=10e-7, n_epochs=100000)

