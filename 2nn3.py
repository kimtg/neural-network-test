import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return np.where(x < 0, 0, 1)

x = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

w1 = np.random.random((3, 5))
w2 = np.random.random((5, 1))

lr = 0.1
for i in range(10000):
    z1 = x.dot(w1)
    a1 = relu(z1)
    z2 = a1.dot(w2)
    a2 = sigmoid(z2)

    delta2 = y - a2
    delta1 = delta2.dot(w2.T)
    delta0 = relu_deriv(z1) * delta1

    w2 += lr * a1.T.dot(delta2)
    w1 += lr * x.T.dot(delta0)

    if i % 1000 == 0:
        print("Error", np.mean(np.abs(delta2)))
