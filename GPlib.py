
import numpy as np
import random
from deap import base, creator, tools, gp
from sklearn.base import BaseEstimator, RegressorMixin
import importlib
import inspect
from functools import partial
import warnings
import GPFunction
import GPmemorize

class GPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, pop_size=100, gen_num=20,constant_range = None,
                 primitive_module=None, genetic_operator_pipline=None, fitness_function=None,fitness_weight= -1,init_mintree_height=1,init_maxtree_height=3,
                 parsimony=0, seed=None,value_log = None,hof_size=1,elitism=False):
        '''
        popsize: int, size of the population
        gen_num: int, number of generations
        primitive_module: module, user-defined primitiveset which should be created in another .py file and imported.
        genetic_operator_pipline: class module, user defined pipline of genetic operations, including selection, reproduction, crossover and mutation
        fitness function: function
        parsimony: float, panelty of size
        seed
        constant_range: None or 2 ele list, e.g., [-1,1] for epheral constant
        fitness_weight: the weight of fitness, usually -1 (min is best) or 1 (max is best)
        init_mintree_height: the tree min height in initial population
        init_maxtree_height: the tree max height in initial population
        '''
        self.pop_size = pop_size
        self.gen_num = gen_num
        self.primitive_module = primitive_module
        self.genetic_operator_pipline = genetic_operator_pipline
        self.fitness_function = fitness_function
        self.parsimony = parsimony
        self.seed = seed
        self.constant_range = constant_range # 一个列表，比如[-1,1]
        self.fitness_weight = fitness_weight
        self.value_log = value_log
        self.init_mintree_height = init_mintree_height
        self.init_maxtree_height = init_maxtree_height
        self.hof_size = hof_size
        self.elitism = elitism

        # 没有value_log的时候提醒创建一个，不然无法加速
        if value_log is None:
            warnings.warn('No value_log, use shared_log = GPmemorize.get_shared_log() to create one',UserWarning)
        if fitness_function is None:
            warnings.warn('No Custom defined fitness function,use MSE instead',UserWarning)
            fitness_weight = -1 # 默认使用MSE作为fitness function的时候，fitness越小越好
        if genetic_operator_pipline is None:
            warnings.warn('No Custom defined Genetic Operations,use default: 3 Tournament, 0.9 std crossover, and 0.2 NodeReplace mutation instead',UserWarning)
    
    def _load_module_functions(self, module_name):
        try:
            mod = importlib.import_module(module_name)
            return {name: func for name, func in inspect.getmembers(mod, inspect.isfunction)}
        except ModuleNotFoundError:
            warnings.warn(f"Module {module_name} not found.")
            return {}

    def _setup_gp(self, X, y):
        constant_range = self.constant_range
        if self.seed is not None:
            np.random.seed(self.seed)
        self.pset = gp.PrimitiveSet("MAIN", X.shape[1])

        # 加载PrimitiveSet
        if self.primitive_module:
            func_dict = self._load_module_functions(self.primitive_module)
        else:
            func_dict  = self._load_module_functions(GPFunction)
        for name, func in func_dict.items():
            arity = func.__code__.co_argcount # PrimitiveFunction的声明数量
            self.pset.addPrimitive(func, arity, name=name)

        # 添加常数
        if constant_range == None:
            pass
        # 检查constant_range是否合规：两个元素的列表，且第一个元素小于第二个元素
        elif len(constant_range) == 2:
            if constant_range[0]<=constant_range[1]:
                self.pset.addEphemeralConstant("rand0", partial(random.uniform, constant_range[0], constant_range[1]))
            else:
                raise ValueError("the 1st in constant_range must be smaller than the 2nd element")
        else:
            raise ValueError("constant_range must be a 2 element list")
        # 创建fitness
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(self.fitness_weight,)) 
        # 创建个体
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) 

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=self.init_mintree_height, max_=self.init_maxtree_height)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
  
        def eval_func(ind):
            try:
                # 计算树在X上的输出
                pred = GPmemorize.compute_tree(ind, pset=self.pset, x=X,shared_log=self.value_log)
                if callable(self.fitness_function):
                    loss = self.fitness_function(pred, y)
                # 使用自定义的fitness function进行衡量
                elif isinstance(self.fitness_function):
                    loss = self.fitness_function(pred, y)
                else:
                    # 用户没有定义的话就默认用MSE函数
                    loss = np.mean((pred - y) ** 2)
                return (loss + self.parsimony * len(ind)*self.fitness_weight,)
            except:
                return (float("inf"),)
        self.toolbox.register("evaluate", eval_func)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self._setup_gp(X, y)
        pop = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(self.hof_size)
        # 初次evaluation
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for _ in range(self.gen_num):
            if self.genetic_operator_pipline == None: # 如果用户没有定义pipline，就用默认设置
                # 锦标赛选择
                offspring = tools.selTournament(pop,len(pop),3)
                offspring = list(map(self.toolbox.clone, offspring))
                # 标准交叉
                for i in range(1, len(offspring), 2):
                    if np.random.rand() < 0.9:
                        offspring[i-1], offspring[i] = gp.cxOnePoint(offspring[i-1], offspring[i])
                        del offspring[i-1].fitness.values, offspring[i].fitness.values
                # 点突变
                for i in range(len(offspring)):
                    if np.random.rand() < 0.2:
                        offspring[i], = gp.mutNodeReplacement(offspring[i])
                        del offspring[i].fitness.values
            else:
                offspring = self.genetic_operator_pipline(pset=self.pset).apply(pop)
            pass
            pass
            pass
            invalids = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalids))
            for ind, fit in zip(invalids, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)

        self._best_ind = hof[0]
        self._compiled_func = gp.compile(expr=self._best_ind,pset=self.pset)
        return self

    def predict(self, X):
        X = np.array(X)
        return np.array([self._compiled_func(*x) for x in X])

    def score(self, X, y):
        y_pred = self.predict(X)
        return -np.mean((y - y_pred) ** 2)
