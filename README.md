# Neural-Net
A simple feedforward neural net using stochastic gradient descent, implemented in python using Numpy. This implementation is meant to be as simple and easy to follow as possible, and is meant for deep learning beginners who want to get an idea of how neural nets work at the implementation level.

## Usage

To try this out, all you have to do is instantiate the FeedForwardSGD class and train it on some appropriate data:

```python
# Two layer neural net with ReLU and sigmoid activation functions, 784x1 size input, 30x1 size hidden layer and 10x1 size output
ff = FeedForwardSGD([784, 30, 10], ["ReLU", "Sigmoid"])
ff.train(xTrain, yTrain)
score = ff.score(xTest, yTest)
```

You can also run a single data instance through the net by calling the forward pass function:

```python
output = ff.forwardPass(xInstance)
```

The file mnist.py includes an example that you can run immediately. It will train a three layer net on the classical MNIST dataset for handwritten digits recognition.
