import numpy as np
import time
from deap import base, creator, tools, gp
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
import GPmemorize
import GPutilities
import multiprocessing
from functools import partial
from tqdm import tqdm
from joblib import Parallel, delayed
import threading
import os
import shutil
from collections import deque

class GPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        pset=None,
        pop_size=100,
        gen_num=20,
        genetic_operator_pipline=None,
        fitness_function=None,
        fitness_weight=-1,        
        parsimony=0.000,
        value_log=None,
        init_mintree_height=1,
        init_maxtree_height=3,
        hof_size=1,
        elitism=False,
        seed=None,
        n_jobs=1,
        verbose=True,
    ):
        """
        pset: PrimitiveSet, use it as DEAP PrimitiveSet \\
        popsize: int, size of the population \\
        gen_num: int, number of generations \\
        genetic_operator_pipline: class module, user defined pipline of genetic operations, including selection, reproduction, crossover and mutation \\
        fitness function: function \\
        parsimony: float, panelty of size \\
        seed: random seeds \\
        value_log: a dict used during evaluation for memorize subtrees result, use shared_log = GPmemorize.get_shared_log() to create one. \\
        fitness_weight: the weight of fitness, usually -1 (min is best) or 1 (max is best) \\
        init_mintree_height: the tree min height in initial population \\
        init_maxtree_height: the tree max height in initial population \\
        hof_size: int, the size of hall-of-frame \\
        elitism: bool, flag to open elitism \\
        verbose: bool, open the log or not. \\
        n_jobs: int, num of pool used
        """
        self.pop_size = pop_size
        self.gen_num = gen_num
        self.pset = pset
        self.genetic_operator_pipline = genetic_operator_pipline
        self.fitness_function = fitness_function
        self.parsimony = parsimony
        self.seed = seed
        self.fitness_weight = fitness_weight
        self.value_log = value_log
        self.init_mintree_height = init_mintree_height
        self.init_maxtree_height = init_maxtree_height
        self.hof_size = hof_size
        self.elitism = elitism
        self.verbose = verbose
        self.n_jobs = n_jobs

        if pset is None:
            raise ValueError("pset is empty, you must have one.")
        # 没有value_log的时候提醒创建一个，不然无法加速
        if value_log is None:
            warnings.warn(
                "No value_log, use shared_log = GPmemorize.get_shared_log() to create one",
                UserWarning)
            time.sleep(1)
        if fitness_function is None:
            warnings.warn(
                "No Custom defined fitness function,use MSE instead", UserWarning)
            time.sleep(1)
            fitness_weight = (-1)  # 默认使用MSE作为fitness function的时候，fitness越小越好
        if genetic_operator_pipline is None:
            warnings.warn(
                "No Custom defined Genetic Operations,use default: 3 Tournament, 0.9 std crossover, and 0.2 NodeReplace mutation instead",
                UserWarning)
            time.sleep(1)


    def _setup_gp(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        # 创建fitness
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(self.fitness_weight,))
        # 创建个体
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "expr",
            gp.genHalfAndHalf,
            pset=self.pset,
            min_=self.init_mintree_height,
            max_=self.init_maxtree_height)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

    def eval_func(self,ind,X,y):
        try:
            # 计算树在X上的输出
            pred = self.compute_tree(
                ind, pset=self.pset, x=X, shared_log=self.value_log
            )
            if callable(self.fitness_function):
                loss = self.fitness_function(pred, y) # 使用自定义的fitness function进行衡量
            else:
                # 用户没有定义的话就默认用MSE函数
                loss = np.mean((pred - y) ** 2)
            return (loss + self.parsimony * len(ind) * self.fitness_weight,)
        except Exception as e:
            print(ind)
            raise e
            return (float("inf"),)
    def log_decorator(self,shared_log, expr_str):
        """
        装饰器：包装原语函数，记录函数名称、输入、输出以及形状信息。
        
        参数:
        - shared_log: 共享日志列表
        - expr_str: 当前调用的表达式字符串（例如 "add(ARG0, ARG1)"）
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                input_values = tuple(args)
                # 生成哈希键，唯一标识一个计算
                key = expr_str
                # 检查共享日志
                if key in shared_log:
                    shared_log[key]["count"] += 1
                    return shared_log[key]["output_value"]  # 直接返回已计算结果
                else:
                    # 调用原函数
                    result = func(*args, **kwargs)
                    shared_log[key] = {
                        "function": expr_str,
                        "input_values": input_values,
                        "output_value": result,
                        "count": 1,
                    }
                    return result
            return wrapper
        return decorator

    def compute_tree(self,expr, pset,x, prefix="ARG",overflow_inf = True,shared_log=None,MIcultuation = True):
        # 初始化栈，用于递归计算每个节点
        stack = deque()
        # 遍历树中的每个节点并递归计算值
        for id, node in enumerate(expr):
            stack.append((node, [], [], id))  # 将节点和空参数列表压入栈
            while len(stack[-1][1]) == stack[-1][0].arity:  # 确保所有子节点都被处理
                prim, args, arg_expressions, id = stack.pop()  # 获取当前节点的原语和参数
                if isinstance(prim, gp.Primitive):
                    # 对于 Primitive 节点，调用相应的原语函数计算结果
                    if shared_log is not None: # 是否开启了share_log功能
                        if arg_expressions:
                            expr_str = f"{prim.name}({', '.join(arg_expressions)})" # 如果有拼好的表达式就直接拿来用
                        else:
                            expr_str = f"{prim.name}({', '.join(map(str, args))})" # 如果没有就重新创建一个
                        decorated_func = self.log_decorator(shared_log, expr_str)(pset.context[prim.name]) # 调用当前的函数
                    else:
                        expr_str = None
                        decorated_func = pset.context[prim.name] # 没开启就直接调用函数
                    try:
                        result = decorated_func(*args)
                    except OverflowError as e:
                        # 对溢出的处理
                        if overflow_inf == True:
                            result = float('inf') # 返回inf
                        else:
                            result = args[0]  # 返回第一个参数作为结果

                elif isinstance(prim, gp.Terminal):
                    # 对于 Terminal 节点，获取数据集中的相应特征值
                    if prefix in prim.name: # 这是个变量
                        if isinstance(x, (np.ndarray)):
                            expr_str = prim.name
                            if x.ndim == 1:
                                result = x  # 直接使用整个数组
                            else:
                                result = x[:, int(prim.name.replace(prefix, ""))] # 把变量对应的值带入运算
                        else:
                            raise ValueError("Datatype should be np.ndarray")
                    else:
                        result = float(prim.value)  # 对于常量终结符，直接使用值
                        expr_str = str(prim.value)
                else:
                    raise Exception("Unsupported primitive type!")

                # 将结果传递给父节点（即将当前节点的结果作为参数传递给上层）
                if not stack:
                    break  # 如果栈为空，表示所有节点都已经计算过
                stack[-1][1].append(result)  # 将计算结果添加到栈顶父节点的参数列表
                stack[-1][2].append(expr_str)  # 记录子表达式

        # 最终返回树的计算结果
        return result

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.X, self.y = X, y  # 确保在类内共享
        self._setup_gp()

        self.toolbox.register("evaluate", self.eval_func,X=X,y=y)

        pop = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(self.hof_size)

        # 初次evaluation
        fitnesses = self.toolbox.map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # 初始化记录
        pop_fit = tools.Statistics(lambda ind: ind.fitness.values)
        pop_size = tools.Statistics(len)  # Object: Pop size
        ind_height = tools.Statistics(
            lambda ind: ind.height
        )  # Object:individual height
        mstats = tools.MultiStatistics(
            fitness=pop_fit, size=pop_size, height=ind_height
        )
        record = mstats.compile(pop)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        mstats.register("num", len)
        logbook = tools.Logbook()
        logbook.header = (
            "gen",
            "time",
            "num",
            "new",
            "fitness",
            "size",
            "height",
            "best_ind_height",
            "best_ind_len",
            "best_ind",
        )
        logbook.chapters["fitness"].header = "min", "avg", "max"
        logbook.chapters["size"].header = "min", "avg", "max"
        logbook.chapters["height"].header = "min", "avg", "max"

        for g in range(self.gen_num):
            start_time = time.perf_counter()
            if self.genetic_operator_pipline is None:  # 如果用户没有定义pipline，就用默认设置
                # 锦标赛选择
                offspring = tools.selTournament(pop, len(pop), 3)
                offspring = list(map(self.toolbox.clone, offspring))
                # 标准交叉
                for i in range(1, len(offspring), 2):
                    if np.random.rand() < 0.9:
                        offspring[i - 1], offspring[i] = gp.cxOnePoint(
                            offspring[i - 1], offspring[i]
                        )
                        del offspring[i - 1].fitness.values, offspring[i].fitness.values
                # 点突变
                for i in range(len(offspring)):
                    if np.random.rand() < 0.1:
                        (offspring[i],) = gp.mutNodeReplacement(offspring[i],self.pset)
                        del offspring[i].fitness.values
            else:
                offspring = self.genetic_operator_pipline(pset=self.pset).apply(pop)

            # 重新衡量结构有改动的个体
            # 并行评估
            invalids = [ind for ind in offspring if not ind.fitness.valid]
            with multiprocessing.Pool(processes=self.n_jobs) as pool:
                fitnesses = pool.map(self.toolbox.evaluate, invalids)
            for ind, fit in zip(invalids, fitnesses):
                ind.fitness.values = fit
            # 精英
            if self.elitism == True:
                offspring = GPutilities.elitism(offspring, hof)
            else:
                pass

            pop[:] = offspring
            hof.update(pop)
            end_time = time.perf_counter()
            time_cost = float(end_time - start_time) * 1000.0
            best_ind = hof[0]
            if self.verbose == True:
                record = mstats.compile(pop)
                logbook.record(
                    gen=g,
                    time=time_cost,
                    num=len(pop),
                    best_ind=str(best_ind),
                    best_ind_len=len(best_ind),
                    best_ind_height=best_ind.height,
                    new=len(invalids),
                    **record,
                )
                print("------------------------------")
                print(logbook.stream)
                

        self._best_ind = hof[0]
        self._best_ind_fitness = float(self._best_ind.fitness.values[0])
        self._compiled_func = gp.compile(expr=self._best_ind, pset=self.pset)
        GPutilities.SaveLogbookToPickle(logbook, self._best_ind_fitness)
        return self

    def predict(self, X):
        X = np.array(X)
        return np.array([self._compiled_func(*x) for x in X])