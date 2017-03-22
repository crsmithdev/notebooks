import numpy as np
import sklearn
import sklearn.datasets
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Network(object):
    
    def __init__(self, shape, reg=0.01):
        inputs = shape[:-1]
        outputs = shape[1:]

        self.weights = np.array([np.random.randn(i, o) for i, o in zip(inputs, outputs)])
        self.biases = [np.random.randn(1, o) for o in outputs]
        self.regularization = reg

    def predict(self, X):
        activations, _ = self.feed_forward(X)
        return np.argmax(activations[-1], axis=1)

    def cost(self, X, y):

        activations, _ = self.feed_forward(X)
        output = activations[-1]
        err = (-y) * np.log(output) - (1 - y) * np.log(1 - output)
        cost = np.sum(err) / len(X)

        return cost

    def feed_forward(self, examples):

        activations = [examples]
        combinations = [examples]

        for weights, bias in zip(self.weights, self.biases):
            combinations.append(np.dot(activations[-1], weights) + bias)
            activations.append(sigmoid(combinations[-1]))

        return activations, combinations

    def backpropagate(self, examples, labels):

        activations, combinations = self.feed_forward(examples)
        delta = activations[-1] - labels
        weight_grads = []
        bias_grads = []

        for i in reversed(range(len(self.weights))):
            weight = self.weights[i]
            
            weight_grads.append(np.dot(activations[i].T, delta) + self.regularization * weight)
            bias_grads.append(np.sum(delta, axis=0))
            
            delta = delta.dot(weight.T) * sigmoid_prime(combinations[i])

        weight_grads.reverse()
        bias_grads.reverse()

        return weight_grads, bias_grads

    def fit(self, examples, labels, iterations=20000, learning_rate=0.01, iter_fn=None):

        for i in range(iterations):
            weight_grads, bias_grads = network.backpropagate(examples, labels)

            self.weights = [w - learning_rate * g for w, g in zip(self.weights, weight_grads)]
            self.biases = [b - learning_rate * g for b, g in zip(self.biases, bias_grads)]

            if iter_fn: iter_fn(i)
         
        return self

X, y = sklearn.datasets.make_moons(200, noise=0.20)
yy = label_binarize([i + 1 for i in y], [1, 2, 3])[:,0:2]

def iter_fn(i):
    if i % 1000 == 0:
        print(i, network.cost(X, yy))

shape = (X.shape[1], 10, 10, yy.shape[1])

network = Network(shape, reg=0.01)
network.fit(X, yy, iter_fn=iter_fn)
predicted = network.predict(X)

print(accuracy_score(y, predicted))