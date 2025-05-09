from GPlib import GPRegressor
import numpy as np
from GeneticOperationPipline import GeneticOperationPipeline
from deap import gp
import random
from sklearn.metrics import mean_absolute_error
from functools import partial
import multiprocessing
import GPmemorize

random.seed(10)
np.random.seed(10)
# 创建一个二维网格的点坐标
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
# 将网格展平成一个 (10000, 2) 的点集
X = np.column_stack([X1.ravel(), X2.ravel()])
y = X[:, 0] ** 2 + np.sin(X[:, 1])


# Protective Div
def protected_div(left, right):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x if isinstance(x, np.ndarray) else float(x)


main_set = gp.PrimitiveSet(
    "MAIN", X.shape[1]
)  # the number of variables are the size of x
main_set.addPrimitive(np.add, 2)
main_set.addPrimitive(np.subtract, 2)
main_set.addPrimitive(np.multiply, 2)
main_set.addPrimitive(np.sin, 1)
main_set.addPrimitive(np.cos, 1)
main_set.addPrimitive(np.tan, 1)
# main_set.addTerminal(np.pi)
main_set.addPrimitive(protected_div, 2, name="div")
# main_set.addEphemeralConstant("rand0", partial(random.uniform, -1, 1))


def mse_fitness(y_train, y_pred):
    return mean_absolute_error(y_train, y_pred)


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows 系统下需要调用
    # 创建 Manager 对象，传递给 MIGP 中的共享 log
    manager = multiprocessing.Manager()
    valuelog = GPmemorize.get_shared_log(manager)  # 在主进程中创建共享日志
    model = GPRegressor(
        gen_num=20,
        pop_size=300,
        pset=main_set,
        genetic_operator_pipline=GeneticOperationPipeline,
        fitness_function=mse_fitness,
        n_jobs=2,
        hof_size=5,
        elitism=True,
        seed=random.seed(10),
        init_mintree_height=2,
        init_maxtree_height=6,
        value_log=valuelog
    )
    model.fit(X, y)
    print(valuelog)