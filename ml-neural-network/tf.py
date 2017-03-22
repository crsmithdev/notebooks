import tensorflow as tf
import sklearn.datasets
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score
import numpy as np

class TFNeuralNetwork(object):

    def __init__(self, shape, regularization=0.01):

        self.input = tf.placeholder(tf.float32, [None, shape[0]])
        self.output = tf.placeholder(tf.float32, [None, shape[-1]])

        self.weights = [tf.Variable(tf.random_normal([i, o])) for i, o in zip(shape[:-1], shape[1:])]
        self.biases = [tf.Variable(tf.random_normal([o])) for o in outputs]

        self.model = self.build_model()
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model,
            labels=self.output))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

    def build_model(self):

        activations = [self.input]

        for weights, bias in zip(self.weights, self.biases):
            activation = tf.sigmoid(tf.matmul(activations[-1], weights) + bias)
            activations.append(activation)

        return activations[-1]

    def predict(self, session, examples):
        return session.run(self.model, feed_dict={self.input: examples})

    def fit(self, session, examples, labels, iterations=10000, iter_fn=None):

        for i in range(iterations):
            _, c = session.run([self.optimizer, self.cost], feed_dict={
                self.input: examples,
                self.output: labels
            })

with tf.Session() as sess:

    examples, labels = sklearn.datasets.make_moons(200, noise=0.20)
    labels = label_binarize([i + 1 for i in labels], [1, 2, 3])[:,0:2]
    network = TFNeuralNetwork((examples.shape[1], 10, 10, labels.shape[1]))

    sess.run(tf.global_variables_initializer())

    network.fit(sess, examples, labels)
    predictions = network.predict(sess, examples)

    print(accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1)))