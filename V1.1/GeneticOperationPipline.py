from deap import gp,base,tools
import random
class GeneticOperationPipeline:
    def __init__(self, pset,mate_rate=0.9, mut_rate=0.08, point_muta_rate=0.01):
        self.toolbox = base.Toolbox()
        self.mate_rate = mate_rate
        self.mut_rate = mut_rate
        self.point_muta_rate = point_muta_rate
        self.pset = pset
        self.toolbox.register("select", tools.selTournament, tournsize=60)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=2, max_=4)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        self.toolbox.register("point_mutate", gp.mutNodeReplacement, pset=self.pset)
        self.toolbox.decorate("mate", gp.staticLimit(key=len, max_value=250))
        self.toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=250))

    def apply(self, pop):
        offspring = self.toolbox.select(pop, len(pop))
        offspring = list(map(self.toolbox.clone, offspring))

        for i in range(1, len(offspring), 2):
            if random.random() < self.mate_rate:
                offspring[i - 1], offspring[i] = self.toolbox.mate(offspring[i - 1],
                                                                   offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < self.mut_rate:
                offspring[i], = self.toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < self.point_muta_rate:
                offspring[i], = self.toolbox.point_mutate(offspring[i])
                del offspring[i].fitness.values
        return offspring