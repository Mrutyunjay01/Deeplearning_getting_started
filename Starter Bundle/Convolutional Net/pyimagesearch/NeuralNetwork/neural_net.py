import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha

        for i in np.arange(0, len(layers)-2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)  # 1 is added becoz of the bias term
            self.W.append(w / np.sqrt(layers[i]))
        # last two layers input need bias term but the output doesn't
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))
        # till now the weights are randomly sampled and normalised
    # now we will define a python magic method '__repr__'
    # which is basically helpful in debugging

    def __repr__(self):
        # construct and return a string that represents
        # the network architecture
        return "NeuralNetwork : {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1-x)

    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)]
        """
        construct our list of output activations for each layer 
        as our data point flows through the network ; the first 
        activation is a special case: it's just the input feature 
        vector itself.
        """
        ##FEEDFORWARD##
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer]) # perform dot product bet the W and X matrix
            out = self.sigmoid(net) # pass it through the activation layer
            A.append(out)  # add it to the list of activations

        ##BACKPROP##
        # 1st step is to calculate the error which is between
        # predcited value - ground truth level
        error = A[-1] - y

        # now we need to apply the delta rule
        # first entry should be the error itself
        D = [error * self.sigmoid_deriv(A[-1])]
        # now iterate through the epochs using the first delta value
        for layer in np.arange(len(A) - 2, 0, -1):
            """
            delta for the current layer is equal to the 
            delta of the prev layer * weight matrix of the current layer
            followed by multiplying the delta by the derivative of the non_linear acitvation
            function of the current layer
            """
            delta = D[-1].dot(self.W[layer].T)
            delta *= self.sigmoid_deriv(A[layer])
            D.append(delta)
        # since we have looped over in reverse order
        # revert back the deltas
        D = D[::-1]
        # now updating the weight matirx based on the deltas collected
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)

        if addBias:
             p = np.c_[p, np.ones((p.shape[0]))]
        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions-targets) ** 2)

        return loss

    def fit(self, X, y, epochs=10, displayUpdate=100):
        X = np.c_[X, np.ones((X.shape[0]))]  # insert a col of 1's as the last entry to add the bias as a trainabe parameter inside the weight matrix
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
            if epoch == 0 or (epoch + 1)% displayUpdate ==0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch = {}, loss={:.7f}".format(epoch + 1, loss))

