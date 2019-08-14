import tensorflow as tf
import cv2
import numpy as np
from feedforward import FeedForwardSGD
from activations import *

debug = False

# load in our dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

if debug:
    cv2.imshow('image', x_train[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # dimensions of each image
    dimensions = (len(x_train[0]), len(x_train[0][0]), len(x_train))
    print("train set dimensions: {}".format(dimensions))

# flatten images
print("flattening images...")
x_train_flatten = []
x_test_flatten = []
y_train_onehot = []
y_test_onehot = []
for i in range(len(x_train)):
    image = x_train[i]
    y_encoded = np.zeros((10,1))
    y_encoded[y_train[i]] = 1

    x_flatten = image.reshape(image.shape[0]*image.shape[1],1)
    x_flatten_norm = x_flatten/np.linalg.norm(x_flatten)

    x_train_flatten.append(x_flatten_norm)
    y_train_onehot.append(y_encoded)
for i in range(len(x_test)):
    image = x_test[i]
    y_encoded = np.zeros((10,1))
    y_encoded[y_train[i]] = 1

    x_flatten = image.reshape(image.shape[0]*image.shape[1],1)
    x_flatten_norm = x_flatten/np.linalg.norm(x_flatten)

    x_test_flatten.append(x_flatten_norm)
    y_test_onehot.append(y_encoded)
print("     flattened images!")

# initialise our FeedForwardSGD net
print("initialising neural net...")
ff = FeedForwardSGD([784,30,10], ["ReLU", "Sigmoid"])
print("     initialised neural net!")

#ff.forwardPass(np.random.randint(2, size=784))

for k in range(30):
    ff.train(x_train_flatten, y_train_onehot, epochs=1)
    print("epoch {}   score {}".format(k, ff.score(x_test_flatten, y_test)))

for i in range(20):

    print("")

    scores = ff.forwardPass(x_test_flatten[i])
    scores = scores/np.sum(scores)

    for j in range(0,10):
        print("{} probability: {}".format(j,scores[j]))

    cv2.imshow('image', x_test[i])
    cv2.waitKey(0)
cv2.destroyAllWindows()
