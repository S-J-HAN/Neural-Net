import numpy as np

def ReLU(x):
    return np.maximum(x, 0)

def Softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def Sigmoid(x):
    return 1/(1+np.exp(-x))

def Tanh(x):
    return np.tanh(x)

def ReLUGradient(x):
    x1 = np.array(x)
    return np.where(x1<0,0,1)

def SigmoidGradient(x):
    return np.exp(-x)/np.square(1 + np.exp(-x))

def TanhGradient(x):
    return 1 - np.square(np.tanh(x))
