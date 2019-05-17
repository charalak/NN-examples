import numpy as np
import matplotlib.pyplot as plt



def forward_propagation(X, W1, b1, W2, b2):
    # tanh activation functiomn
    Z =  1/ (1+ np.exp( -(X.dot(W1)+b1)))
    a = Z.dot(W2) + b2
    Y = 1 / (1 + np.exp(-a))
    # softmax activation for the output
    return Y,  Z


def predict(X, W1,b1,W2,b2):
    Y , _ = forward_propagation(X, W1, b1, W2, b2)
    return np.round(Y)


def derivative_w2(Z, T, Y):
    der_w2 = Z.T.dot(T-Y)
    return der_w2


def derivative_b2(T, Y):
    der_b2 = (T - Y).sum()
    return der_b2


def derivative_w1(X, Z, T, Y, W2):
    Zprime = np.outer(T-Y, W2) * (1- Z * Z)
    der_w1 = X.T.dot(Zprime)
    return der_w1


def derivative_b1(Z, T, Y, W2):
    der_b1 = np.outer(T-Y, W2) * (1 - Z * Z)
    return der_b1.sum(axis=0)

def cost_logloss(T, Y):
    return np.sum(T*np.log(Y) + (1-T)*np.log(1-Y))
    # tot = T * np.log(Y)
    # return tot.sum()

def test_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])

    W1 = np.random.randn(2, 4)
    b1 = np.random.randn(4)
    W2 = np.random.randn(4)
    b2 = np.random.randn(1)

    LL = []
    eta = 0.0005
    regularization = 0.
    last_error_rate = None

    for i in range(100000):
        pY, Z = forward_propagation(X, W1, b1, W2, b2)
        ll = cost_logloss(Y, pY)

        prediction = predict(X, W1, b1, W2, b2)
        er = np.abs(prediction - Y).mean()

        if er != last_error_rate:
            last_error_rate = er
            print("Error rate = {}".format(er))
            print("True: {}".format(Y))
            print("predicted: {}".format(prediction))

        if LL and ll < LL[-1]:
            print("early exit")
            break
        LL.append(ll)

        W2 += eta * (derivative_w2(Z, Y, pY) - regularization * W2)
        b2 += eta * (derivative_b2(Y, pY) - regularization * b2)
        W1 += eta * (derivative_w1(X, Z, Y, pY, W2) - regularization * W1)
        b1 += eta * (derivative_b1(Z, Y, pY, W2) - regularization * b1)

        if i% 10000 ==0:
            print(ll)

    print("Classification rate = {}".format(np.mean(prediction == Y)))
    plt.plot(LL)
    plt.show()


def test_donut():
    N = 1000
    R_inner = 5
    R_outer = 10

    R1 = np.random.randn(int(N/2)) + R_inner

    theta = 2 * np.pi * np.random.randn(int(N/2))
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(int(N/2)) + R_outer
    theta = 2 * np.pi * np.random.randn(int(N/2))
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])
    Y = np.array([0]*int(N/2) + [1]*int(N/2))

    n_hidden = 8
    W1 = np.random.randn(2, n_hidden)
    b1 = np.random.randn(n_hidden)
    W2 = np.random.randn(n_hidden)
    b2 = np.random.randn(1)

    LL = []
    eta = 0.0005
    regularization = 0.2
    last_error_rate = None

    for i in range(160000):
        pY, Z = forward_propagation(X, W1, b1, W2, b2)
        ll = cost_logloss(Y, pY)

        prediction = predict(X, W1, b1, W2, b2)
        er = np.abs(prediction - Y).mean()

        LL.append(ll)

        W2 += eta * (derivative_w2(Z, Y, pY) - regularization * W2)
        b2 += eta * (derivative_b2(Y, pY) - regularization * b2)
        W1 += eta * (derivative_w1(X, Z, Y, pY, W2) - regularization * W1)
        b1 += eta * (derivative_b1(Z, Y, pY, W2) - regularization * b1)

        if i % 100 ==0:
            print('ll = {} ; cr = {}'.format(ll, 1-er))

    print("Classification rate = {}".format(np.mean(prediction == Y)))
    plt.plot(LL)
    plt.show()



if __name__ == '__main__':
    # test_xor()
    test_donut()