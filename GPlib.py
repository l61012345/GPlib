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

    def exist_cache(self, cache_dir="gp_cache"):
        """
        检查缓存文件夹是否存在
        """
        if os.path.exists(cache_dir):
            warnings.warn('Cache already existsm,try to clean')

    def _setup_gp(self):
        self.exist_cache()  # 检查缓存
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
            pred = GPmemorize.compute_tree(
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

    def parallel_eval(self,individuals, n_jobs=4):
    # 每个进程用一个自己的缓存
        thread_local = threading.local()
        def wrapped_eval(ind):
            if not hasattr(thread_local, "cache"):
                thread_local.cache = {}
            return self.eval_func(ind, thread_local.cache)
        return Parallel(n_jobs=n_jobs)(delayed(wrapped_eval)(ind) for ind in individuals)
    

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self._setup_gp()
        parallel_eval = partial(self.eval_func, X=X, y=y)
        self.toolbox.register("evaluate", parallel_eval)
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

        with multiprocessing.Pool(processes=self.n_jobs) as pool:
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
                invalids = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = list(tqdm(pool.map(parallel_eval, invalids),total=len(invalids)))
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
                best_ind = tools.selBest(hof, 1)[0]
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
                # 清理日志的机制
                if g%2==0:
                    if self.value_log is not None:
                        self.value_log = GPmemorize.clean_log(self.value_log,100)
                    
        pool.close()
        pool.join()
        self._best_ind = hof[0]
        self._best_ind_fitness = float(self._best_ind.fitness.values[0])
        self._compiled_func = gp.compile(expr=self._best_ind, pset=self.pset)
        GPutilities.SaveLogbookToPickle(logbook, self._best_ind_fitness)
        return self

    def predict(self, X):
        X = np.array(X)
        return np.array([self._compiled_func(*x) for x in X])