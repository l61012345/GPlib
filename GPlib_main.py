from GPlib import GPRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import GeneticOperationPipline
from deap import gp
import random
from sklearn.metrics import mean_absolute_error
from functools import partial
X = np.random.rand(100, 2)
y = X[:, 0] ** 2 + np.sin(X[:, 1])

# Protective Div
def protected_div(left, right):
    with np.errstate(divide='ignore',invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x if isinstance(x, np.ndarray) else float(x)

main_set = gp.PrimitiveSet("MAIN", X.shape[1])  # the number of variables are the size of x
main_set.addPrimitive(np.add, 2)
main_set.addPrimitive(np.subtract, 2)
main_set.addPrimitive(np.multiply, 2)
main_set.addPrimitive(np.sin, 1)
main_set.addPrimitive(np.cos, 1)
main_set.addPrimitive(np.tan,1)
main_set.addTerminal(np.pi)
main_set.addPrimitive(protected_div, 2, name="div")
main_set.addEphemeralConstant("rand0", partial(random.uniform, -1, 1))

def mse_fitness(y_train,y_pred):
    return mean_absolute_error(y_train, y_pred)

if __name__ =="__main__":
    model = GPRegressor(pset=main_set)
    model.fit(X,y)