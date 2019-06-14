import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(64)

alpha = 0.01

# activation fucntion
def step(x):
    if x < 0:
        return 0
    else:
        return 1

# forward network
def forward(x, w):
    # Calc weighted sum
    out = np.sum(w[:-1]*x) + w[-1]
 
    # activate function
    y = step(out)
    
    return y

# Learning
def train(x, w, y, t):
    # update weights
    for i in range(len(w)-1):
        w[i] += alpha * (t - y) * x[i]
    
    w[-1] += alpha * (t -y)

# Evaluate
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


if __name__ == '__main__':
    
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    w = np.random.rand(3) # w[-1] is bias weight
    t = np.array([0, 0, 0, 1])
    #t = np.array([0, 1, 1, 1])
    #t = np.array([0, 1, 1, 0])
    
    err = []
    for epoch in range(100):
        # training
        y_pred = []
        for i in range(len(data)):
            y = forward(data[i], w)
            train(data[i], w, y, t[i])
            y_pred.append(y)
        
        err.append(mean_squared_error(np.array(y_pred), t))
        print('Epoch{}:, Pred:{}, T:{}'.format(epoch, y_pred, t)) 
    
    # MSE graph
    plt.plot(err)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.show()
