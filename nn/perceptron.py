import numpy as np

# AND gate
def AND(x1, x2):
    x = np.array([x1, x2])  # Input data
    w = np.array([0.5, 0.5])    # Weights
    b = -0.7    # bias

    # Calc weighted sum
    out = np.sum(w*x) + b
    
    # Step function
    if out <= 0:
        return 0, out
    else:
        return 1, out

    
if __name__ == '__main__':
    
    data = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for data in data:
        y, out = AND(data[0], data[1])
        print(data, ':', y, '|', out)
