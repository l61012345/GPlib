import numpy as np

def add(x,y):
    return np.add(x,y)

def substract(x,y):
    return np.subtract(x,y)

def multiply(x,y):
    return np.multiply(x,y)

def protected_div(left, right):
    with np.errstate(divide='ignore',invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x if isinstance(x, np.ndarray) else float(x)

def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

def tan(x):
    return np.tan(x)