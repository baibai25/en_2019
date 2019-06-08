import numpy as np
np.random.seed(64)

# build 3 layer neural network
def build_network():
    network =  {}
    network['w1'] = np.random.rand(2, 3)
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['w2'] = np.random.rand(3)
    network['b2'] = np.array([0.1])
    #print(network)
    return network

# sigmoid fucntion
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# forward
def forward(network, x):
    # weights and bias
    w1, w2 = network['w1'], network['w2']
    b1, b2 = network['b1'], network['b2']

    # input -> hidden
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)

    # hidden -> output
    a2 = np.dot(z1, w2) + b2
    y = sigmoid(a2)

    return y

    
if __name__ == '__main__':
    network = build_network()
    
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)
 
