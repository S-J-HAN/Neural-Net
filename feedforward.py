import numpy as np
from activations import *
from copy import deepcopy

class FeedForwardSGD:

    def __init__(self, layerSizes, activations):

        assert(len(layerSizes)-1==len(activations))

        self.layerSizes = layerSizes
        self.activations = activations

        inputLayerDims = layerSizes[0]
        outputLayerDims = layerSizes[-1]

        weights = []
        biases = []
        for i in range(0, len(layerSizes)-1):
            weights.append(np.random.randn(layerSizes[i+1], layerSizes[i])*0.01)
            biases.append(np.zeros((layerSizes[i+1],1)))

        self.weights = weights
        self.biases = biases


    def forwardPass(self, x, returnInternals=False):

        assert(len(x.shape)==2 and x.shape[1]==1, "Your input vector has shape {} when it must have shape (n,1)".format(x.shape))

        xEval = x

        z = []
        a = []

        for i in range(0, len(self.layerSizes)-1):
            xEval = np.matmul(self.weights[i], xEval)
            xEval += self.biases[i]
            z.append(xEval)

            # Push through activation function
            if self.activations[i] == "ReLU":
                xEval = ReLU(xEval)
            elif self.activations[i] == "Softmax":
                xEval = Softmax(xEval)
            elif self.activations[i] == "Sigmoid":
                xEval = Sigmoid(xEval)
            elif self.activations[i] == "Tanh":
                xEval = Tanh(xEval)

            a.append(xEval)


        if returnInternals:
            return xEval, z, a
        else:
            return xEval

    def score(self, xTest, yTest):

        predictions = []
        for x in xTest:
            p = self.forwardPass(x)
            predictions.append(np.argmax(p))

        accuracy = np.sum((np.array(predictions)==np.array(yTest)))/len(predictions)

        return accuracy

    def train(self, trainX, trainY, costFunction="MSE", batchSize=1, epochs=1, learningRate=0.01):

        for j in range(epochs):

            # A single iteration through the entire train set
            for i in range(0, len(trainX), 1):

                x = trainX[i:i+batchSize]
                y = trainY[i:i+batchSize]

                Z = []
                for xn in x:
                    Zn, zList, aList = self.forwardPass(xn, returnInternals=True)
                    Z.append(Zn)

                if costFunction == "MSE": # Mean Squared Error
                    dC = Z[0] - y[0]
                elif costFunction == "CE": # Cross Entropy NON FUNCTIONAL
                    dZ = -1/batchSize * np.sum(y/Z)

                # Backprop through layers
                prevdz = None
                for j in range(len(self.layerSizes)-2, -1, -1):

                    w = self.weights[j]
                    b = self.biases[j]
                    z = zList[j]
                    a = aList[j]
                    if j > 0:
                        preva = aList[j-1]
                    else:
                        preva = x[0]
                    activation = self.activations[j]

                    if activation == "Sigmoid":
                        da = SigmoidGradient(z)
                    elif activation == "ReLU":
                        da = ReLUGradient(z)
                    elif activation == "Tanh":
                        da = TanhGradient(z)

                    if j == len(self.layerSizes)-2:
                        dz = np.multiply(dC, da)
                    else:
                        dz = np.multiply(np.matmul(self.weights[j+1].T, prevdz), da)

                    dw = np.matmul(dz, preva.T)
                    prevdz = dz

                    # update weights and biases
                    self.weights[j] -= learningRate*dw
                    self.biases[j] -= learningRate*dz
