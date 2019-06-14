import numpy as np
np.random.seed(64)

# build 3 layer neural network
def build_network():
    # 2 -> 3 -> 1
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
    
    '''
    # Not using numpy.dot *comment out 'return y'
    # input -> hidden
    zz1 = []
    for i in range(3):  # number of hidden units
        aa1 = 0
        for j in range(2):  # number of input units
            aa1 += x[j] * w1[j][i]
        
        aa1 += b1[i]
        zz1.append(sigmoid(aa1))
    
    # hidden -> output
    aa2 = 0
    for i in range(3):
        aa2 += zz1[i] * w2[i]   
    
    aa2 += b2
    yy = sigmoid(aa2)
    
    print(z1, y) 
    print(zz1, yy)
    '''


if __name__ == '__main__':
    network = build_network()
    
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)
 
