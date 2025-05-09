import operator, random, numpy, time,decimal
from datetime import datetime
from functools import partial
import pandas as pd
import numpy as np
from deap import gp, creator, tools, base
import GPutilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,root_mean_squared_error
import multiprocessing

random.seed(10)
np.random.seed(10)

x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
# 将网格展平成一个 (10000, 2) 的点集
X = np.column_stack([X1.ravel(), X2.ravel()])
y = X[:, 0] ** 2 + np.sin(X[:, 1])
current_time = datetime.now().strftime("%Y%m%d_%H%M%S") # local time for log

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


# primitiveset
global main_set
main_set = gp.PrimitiveSet("MAIN", X.shape[1])  # the number of variables are the size of x
main_set.addPrimitive(np.add, 2)
main_set.addPrimitive(np.subtract, 2)
main_set.addPrimitive(np.multiply, 2)
main_set.addPrimitive(np.sin, 1)
main_set.addPrimitive(np.cos, 1)
main_set.addPrimitive(np.tan,1)
#main_set.addPrimitive(np.min,1)
main_set.addTerminal(np.pi)
main_set.addPrimitive(protected_div, 2, name="div")
#main_set.addEphemeralConstant("rand0", partial(random.uniform, -1, 1))

# Create individual
# Creates the attribute fitness of an individual, the base class is Fitness
# since it is as small as possible it is the result calculated by the fitness func multiplied by -1,
# or more than one fitness weights if multiple targets are considered
creator.create("FitnessMin", base.Fitness, weights=(-1,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=main_set, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=main_set)

def evalSymbReg(individual,x_train,y_train, parsimony = 0):
    try:
        func = gp.compile(expr=individual,pset=main_set)
        #y_pred = np.array([func(*x) for x in x_train])
        y_pred = func(*x_train.T)
        print(y_pred)
        # 如果func是常数函数，结果则是一个标量，需要广播出去
        if np.isscalar(y_pred) or (isinstance(y_pred, np.ndarray) and y_pred.size == 1):
            y_pred = np.full_like(y, fill_value=float(y_pred))
        raw_fitness = mean_absolute_error(y_train, y_pred)
        if parsimony != None:
            return (raw_fitness + parsimony * len(individual)),
        else:
            return raw_fitness,
    except Exception as e:
        #print(f"Compilation failed: {e}")
        #print('Error Indiv:', individual)
        return float("Inf"),

# Evolmethod:evalSymbReg
toolbox.register("evaluate_source", evalSymbReg,x_train=X,y_train=y,parsimony = 0)

ind = gp.PrimitiveTree.from_string('subtract(multiply(ARG1, ARG0), sin(subtract(ARG0, ARG0)))',pset=main_set)
fitness = toolbox.evaluate_source(ind)
print(fitness)
