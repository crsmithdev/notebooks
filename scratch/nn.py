import numpy as np
import sklearn
import sklearn.datasets
from sklearn.preprocessing import label_binarize

X, y = sklearn.datasets.make_moons(200, noise=0.20)

n_examples = len(X)
n_features = 2
n_labels = 2
alpha = 0.01
lambda_ = 0.01

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    yy = label_binarize([i + 1 for i in y], [1, 2, 3])[:,0:2]

    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)

    err = (-yy) * np.log(a2) - (1 - yy) * np.log(1 - a2)
    cost = np.sum(err) / n_examples

    return cost

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    z1 = x.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)

    return np.argmax(a2, axis=1)

class Network(object):
    
    def __init__(self, n_features, n_hidden, n_labels):
        self.weights = [
            np.random.randn(n_features, n_hidden),
            np.random.randn(n_hidden, n_labels)
        ]
        self.biases = [
            np.random.randn(1, n_hidden),
            np.random.randn(1, n_labels)
        ]

def train(nn_hdim, num_passes=20000, print_loss=False):

    np.random.seed(0)
    W1 = np.random.randn(n_features, nn_hdim)# / np.sqrt(n_features)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, n_labels)# / np.sqrt(nn_hdim)
    b2 = np.zeros((1, n_labels))

    model = {}
    network = Network(n_features, nn_hdim, n_labels)
    print(network)

    for i in range(0, num_passes):

        z1 = X.dot(W1) + b1
        a1 = sigmoid(z1)
        z2 = a1.dot(W2) + b2
        probs = sigmoid(z2)

        delta3 = probs
        delta3[range(n_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * sigmoid_prime(z1)#(1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        dW2 += lambda_ * W2
        dW1 += lambda_ * W1

        W1 += -alpha * dW1
        b1 += -alpha * db1
        W2 += -alpha * dW2
        b2 += -alpha * db2

        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if print_loss and i % 1000 == 0:
          print("Iteration %i: %f" %(i, calculate_loss(model)))

    return model

model = train(6, print_loss=True)
r = predict(model, X)

from sklearn.metrics import accuracy_score
print(accuracy_score(y, predict(model, X)))