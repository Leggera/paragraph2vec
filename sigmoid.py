import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_grad(f):
    #the input f should be the sigmoid function value of your original input x.

    return f * (1 - f)

