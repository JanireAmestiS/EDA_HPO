import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from EDAspy.optimization import EBNA

import matplotlib.pyplot as plt

from typing import Union, List
import time

from problems import Ensemble_Problem
import torch

class information_frentes():
    def __init__(self):
        super().__init__()
        self.information = []
inff = information_frentes()

class EBNAmod(EBNA):
    def __init__(self,
                 size_gen: int,
                 max_iter: int,
                 dead_iter: int,
                 n_variables: int,
                 possible_values: Union[List, np.array],
                 frequency: Union[List, np.array],
                 alpha: float = 0.5,
                 elite_factor: float = 0.4,
                 disp: bool = True,
                 parallelize: bool = False,
                 init_data: np.array = None,
                 information_frentes = inff):
        super().__init__(size_gen=size_gen, max_iter=max_iter, dead_iter=dead_iter,
                         n_variables=n_variables,possible_values=possible_values,frequency=frequency, alpha=alpha, 
                         elite_factor=elite_factor, disp=disp, parallelize=parallelize, init_data=init_data)
        self.size_gen = size_gen
        self.information_frentes = information_frentes
        
    def _check_generation_no_parallel(self, objective_function: callable):
        global _codification, _problem
        dic_inf = {'fronte':[], 'crowding':[], 'solution':[]}

        individuals = []
        for x in self.generation:
            ind = Individual()
            ind._X = x
            individuals.append(ind)
        
        algo = NSGA2(pop_size=self.size_gen)
        algo.setup(_problem)
        eva = Evaluator()
        pop = Population(individuals=individuals)
        algo.evaluator.eval(_problem,pop,algorithm=algo)
        
        r = RankAndCrowding()
        _problem.evaluate
        sol = r.do(_problem,pop,n_survive=self.size_gen,algorithm=algo)
        solution = []
        frontes = [sol[x].get('rank') for x in range(len(sol))]
        crowding = [sol[x].get('crowding') for x in range(len(sol))]

        for elem in sol:
            solution.append(elem.X)
            
        dic_inf['fronte'] = frontes
        dic_inf['crowding'] = crowding
        dic_inf['solution'] = solution
        
        self.information_frentes.information.append(dic_inf)
        
        torch.save(self.information_frentes.information, 'info_frentes.pt')
        
        evaluations = [sol[x].get('rank') for x in range(len(sol))]
        
        max_loss_sol = np.max([sol[i].F[0] for i,e in enumerate(evaluations)])
        min_loss_sol = 0
        evaluations2 = [(x / (len(evaluations) - 1)) * (max_loss_sol - min_loss_sol) + min_loss_sol for x in evaluations]
        
        for i in range(len(evaluations)):
            evaluations2[i] += sol[i].F[0]
            
        self.evaluations = np.array(evaluations2)
        
    
class contador_generaciones():
    def __init__(self):
        super().__init__()
        self.generacion = 0
        self.individuo = 0
        self.df = []
        self.frontes = []
    def aumentar_individuo(self):
        if self.individuo + 1 >=50:
            self.individuo = 0
            self.generacion += 1
        else:
            self.individuo += 1

contador = contador_generaciones()

def define_initial_frequency(possible_values):
    """Define la frecuencia inicial para los posibles valores de cada variable."""
    frequency = []
    for values in possible_values:
        frequency.append([1/(len(values))] * (len(values)))
    return frequency

def _eda_ensemble_mod(pop_size = 50, n_gen = 7, dead_iter = 7, contador = contador):
    global _problem
    
    variables = ['conv2D_layers', 'amount_filters','residual', 'pooling','kernel_size','strides', 'learning_rate', 'batch_size', 'optimizer', 'amount_neurons','linear_layers']
    n_variables = len(variables)-2
    problem = Ensemble_Problem(contador, n_obj=2,n_constr=0, n_var = n_variables)    
    _problem = problem
    
    possible_values = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],[0,1,2], [0,1,2],[0], [0], [0,1,2,3], [0,1,2,3,4], [0,1,2], [0,1,2,3,4,5,6,7,8,9,10]]
    freqs = define_initial_frequency(possible_values)


    ebna = EBNAmod(size_gen=pop_size, max_iter=n_gen, dead_iter=dead_iter, n_variables=n_variables, alpha=0.5,
        possible_values=possible_values, frequency=freqs, information_frentes = inff)
    ebna_result = ebna.minimize(None, False)
    
    torch.save(ebna, 'ebna_model.pt')
    torch.save(ebna_result, 'ebna_result.pt')
    
    return ebna_result.best_ind,ebna_result.best_cost


best_ind, best_cost = _eda_ensemble_mod()
