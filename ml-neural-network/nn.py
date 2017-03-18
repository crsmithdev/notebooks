import numpy as np
import sklearn
import sklearn.datasets
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score

X, y = sklearn.datasets.make_moons(200, noise=0.20)

n_examples = len(X)
n_features = 2
n_hidden = 6
n_labels = 2
alpha = 0.01
lambda_ = 0.01

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))



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

    def predict(self, X):
        a, _ = self._feed_forward(X)
        return np.argmax(a[-1], axis=1)

    def cost(self, X):
        yy = label_binarize([i + 1 for i in y], [1, 2, 3])[:,0:2]

        a, _ = self._feed_forward(X)
        an = a[-1]

        err = (-yy) * np.log(an) - (1 - yy) * np.log(1 - an)
        cost = np.sum(err) / n_examples

        return cost

    def _feed_forward(self, X):
        a = [X]
        z = [X]

        for w, b in zip(self.weights, self.biases):
            z.append(np.dot(a[-1], w) + b)
            a.append(sigmoid(z[-1]))

        return a, z

    def fit(self, X, num_passes=20000):

        W1 = self.weights[0]
        W2 = self.weights[1]
        b1 = self.biases[0]
        b2 = self.biases[1]
        yy = label_binarize([i + 1 for i in y], [1, 2, 3])[:,0:2]

        for i in range(0, num_passes):

            activations, combinations = self._feed_forward(X)
            a0, a1, a2 = activations
            _, z1, z2 = combinations

            delta_w = []
            delta_b = []

            #iterator = zip(
            #    reversed(activations[:-1]),
            #    reversed(combinations[:-1]),
            #    reversed(self.weights),
            #    reversed(self.biases)
            #)

            #delta = a2 - yy
            delta3 = a2 - yy

            #for a, z, w, b in iterator:
            #    dw = np.dot(a.T, delta) + lambda_ * w
            #    db = np.sum(delta, axis=0)
            #    delta = delta.dot(w.T) * sigmoid_prime(z)

                #delta_w.append(dw)
                #delta_b.append(db)

            delta = a2 - yy

            for j in reversed(range(len(self.weights))):
                a = activations[j]
                z = combinations[j]
                w = self.weights[j]
                b = self.biases[j]

                dw = np.dot(a.T, delta) + lambda_ * w
                db = np.sum(delta, axis=0)
                delta = delta.dot(w.T) * sigmoid_prime(z)

                delta_w.append(dw)
                delta_b.append(db)
                

            dW2 = np.dot(a1.T, delta3)
            db2 = np.sum(delta3, axis=0)
            delta2 = delta3.dot(W2.T) * sigmoid_prime(z1)

            dW1 = np.dot(a0.T, delta2)
            db1 = np.sum(delta2, axis=0)

            dW2 += lambda_ * W2
            dW1 += lambda_ * W1

            delta_w.reverse()
            delta_b.reverse()

            #delta_w = reversed(delta_w)
            #delta_b = reversed(delta_b)

            dW1 = delta_w[0]
            dW2 = delta_w[1]
            db1 = delta_b[0]
            db2 = delta_b[1]

            W1 += -alpha * dW1
            b1 += -alpha * db1
            W2 += -alpha * dW2
            b2 += -alpha * db2

            self.weights = [W1, W2]
            self.biases = [b1, b2]

            if i % 1000 == 0:
                print("Iteration %i: %f" %(i, self.cost(X)))
                #print(dW2)
                #print(delta_w[1])

network = Network(n_features, n_hidden, n_labels)
network.fit(X)

print(accuracy_score(y, network.predict(X)))