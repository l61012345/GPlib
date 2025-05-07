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
#main_set.addTerminal(np.pi)
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
toolbox.register("select", tools.selTournament, tournsize=60)
# Genetic Operator
toolbox.register("mate", gp.cxOnePoint)
# Mutation
# Defining mutation with insertion of full-generated subtrees of depth (min,max)
toolbox.register("expr_mut", gp.genFull, min_=2, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=main_set)
toolbox.register("point_mutate",gp.mutNodeReplacement, pset=main_set)
#toolbox.register("shrink_mutate",gp.mutShrink)
# bloat control 
toolbox.decorate("mate", gp.staticLimit(key=len, max_value=250))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=250))
#toolbox.decorate("shrink_mutate", gp.staticLimit(key=len, max_value=250))

# toolbox.decorate("mate", gp.staticLimit(operator.attrgetter('height'), max_value=20))
# toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter('height'), max_value=20))
# toolbox.decorate("shrink_mutate", gp.staticLimit(operator.attrgetter('height'), max_value=20))

# source task
def source_task(pop_size = 300, hof_size = 5, mate_rate = 0.9, muta_rate = 0.08,point_muta_rate=0.01, shrink_muta_rate = 0.01, gen_num = 20):
    global current_time
    best_ind_fitness = 0
    # initialize the population
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(hof_size)  # elitism pool

    # initialize the timer
    start_time = time.perf_counter()
    # First evaluation
    fitnesses = toolbox.map(toolbox.evaluate_source, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Initalize the Stastics (which takes more time)
    pop_fit = tools.Statistics(
        lambda ind: ind.fitness.values
    )  # Object: fitness
    pop_size = tools.Statistics(len)  # Object: Pop size
    ind_height = tools.Statistics(lambda ind: ind.height)  # Object:individual height
    # do avg,std,min,max on above Objects
    mstats = tools.MultiStatistics(fitness=pop_fit, size=pop_size, height=ind_height)
    record = mstats.compile(pop)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    mstats.register("num", len)

    # Logbook related data
    logbook_source = tools.Logbook()
    logbook_source.header = "gen", "time", "num", "invalid", "fitness", "size", "height",'best_ind'
    logbook_source.chapters["fitness"].header = "min", "avg", "max"
    logbook_source.chapters["size"].header = "min", "avg", "max"
    logbook_source.chapters["height"].header = "min", "avg", "max"

    # Start Evolution process
    for g in range(gen_num):
        # Selection
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))  # intermediate pop

        # Apply crossover and mutation on the offspring
        for i in range(1, len(offspring), 2):
            if random.random() < mate_rate:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                                offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < muta_rate:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values
                
        for i in range(len(offspring)):
            if random.random() < point_muta_rate:
                offspring[i], = toolbox.point_mutate(offspring[i])
                del offspring[i].fitness.values

        # for i in range(len(offspring)):
        #     if random.random() < shrink_muta_rate:
        #         offspring[i], = toolbox.shrink_mutate(offspring[i])
        #         del offspring[i].fitness.values


                    
        # Evaluation on offsprings
        # If crossovers and mutations have occurred, 
        # the fitness of the individual is recalculated to save computation
        # 计算适应度（并行）
        invalids = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate_source, invalids)
        for ind, fit in zip(invalids, fitnesses):
            ind.fitness.values = fit

        # Elitism
        offspring = GPutilities.elitism(offspring, hof)

        pop = offspring # Update the pop
        tools.HallOfFame.update(hof, pop)  # Update the elitism pool
        end_time = time.perf_counter()  # record the timestamp for the end


        # add record
        best_ind = tools.selBest(hof, 1)[0]
        best_ind_fitness = float(best_ind.fitness.values[0])
        record = mstats.compile(pop)
        logbook_source.record(
            gen=g,
            time=float(end_time - start_time) * 1000.0,
            num=len(pop),
            best_ind = str(best_ind),
            invalid=len(invalids),
            **record
        )
        print('------------------------------')
        print(logbook_source.stream)

    # best indiv in one generation
    '''
        print(
            "Best individual is %s, Best results are %s"
            % (best_ind, best_ind.fitness.values)
        )
    '''
    # save the log files
    GPutilities.SaveLogbookToPickle(logbook_source, best_ind_fitness,mark='source')
    pool.close()
    pool.join()
    return hof,pop,best_ind,best_ind_fitness

if __name__ == "__main__":
    # Multi processors
    run_num = 1 # numbers for run the algorithms

    for _ in range(run_num):
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
        hof,pop,best_ind,best_ind_fitness = source_task(gen_num=20)
        toolbox.unregister('map')
        