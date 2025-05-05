from functools import partial
import timeit
import numpy as np
from deap import gp, creator, tools, base
import GPmemorize

if __name__=="__main__":
    # 一个例子
    # 创建基集
    global main_set
    main_set = gp.PrimitiveSet("MAIN", 1)  # 名称和变量数
    # 创建基函数集
    main_set.addPrimitive(np.add, 2)  # 加法,输入数为2
    main_set.addPrimitive(np.subtract, 2)
    main_set.addPrimitive(np.multiply, 2)

    # 初始化个体
    # 创建个体的属性fitness，基类是Fitness，由于是越小越好因此是fitness func计算出来的结果乘以-1，如果考虑多目标则有多个fitness weights
    creator.create("FitnessMin", base.Fitness, weights=(-1,))
    # 创建GP的个体，类型为PrimitiveTree，每个个体有一个attribute-fitness 通过FitnessMin得到/Fitness越小越好
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=main_set, min_=4, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=main_set)

    x_values = np.linspace(-1,1,100)
    y_source_values = np.array([x**4 + x**3 + x**2 + x for x in x_values])
    shared_log = GPmemorize.get_shared_log()

    indiv1 = toolbox.individual()
    indiv2 = toolbox.individual()

    #indiv1 = gp.PrimitiveTree.from_string("add(ARG0,ARG1)", pset=main_set)
    #indiv2 = gp.PrimitiveTree.from_string("add(ARG0,ARG1)", pset=main_set)
    print(indiv1)
    value1 = GPmemorize.compute_tree(indiv1, pset=main_set, x=x_values,shared_log=shared_log)
    t1 =  timeit.timeit(
        partial(GPmemorize.compute_tree, indiv1, pset=main_set, x=x_values,shared_log = None), number=10
    )
    compiled_func = gp.compile(indiv1, pset=main_set)
    value1_lambda = list(map(lambda x: compiled_func(x), x_values))
    t2 = timeit.timeit(
        partial(lambda: list(map(lambda x: compiled_func(x), x_values))), number=10
    )
    if list(value1)==value1_lambda:
        print('pass')
    print(value1)
    print("-----------------------")
    print(indiv2)
    value2 = GPmemorize.compute_tree(indiv2, pset=main_set, x=x_values,shared_log = shared_log)
    print(value2)
    print(x_values)
    print(t1, t2)