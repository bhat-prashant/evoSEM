#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"
import warnings
import pickle
warnings.simplefilter("ignore")

import abc
import random
import warnings
from deap import creator, base, tools, algorithms
from sklearn.base import BaseEstimator, TransformerMixin

random.seed(10)
import numpy as np

np.random.seed(10)
from itertools import cycle
from util import *
import pandas as pd


class AllRelations(BaseEstimator, TransformerMixin, metaclass=abc.ABCMeta):

    def __init__(self, generation=5, pop_size=20, mutation_rate=0.3,
                 crossover_rate=0.7):
        """ Main class for creating SEM model

        :param generation: int,
            number of generations to be run during evolution
        :param pop_size: int
            number of individuals in the initial population
        :param mutation_rate: float
            Percentage of individuals in the population to be mutated during evolution
        :param crossover_rate: float
            Percentage of individuals in the population to be mated during evolution
        """

        self._generation = generation
        self._pop_size = pop_size
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._variables = ['x1', 'x2', 'x3', 'x4', 'x5']
        self._concepts = ['c1', 'c2', 'c3']
        self._fit_indices = ["cfi","tli", "aic", "bic", "rmsea"]
        self._pop = None
        self._toolbox = None

    def _setup_toolbox(self):
        """ Sets up DEAP toolbox for evolution

        :return: None
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            creator.create('FitnessMulti', base.Fitness, weights=(1.0, 1.0, 0.5, 0.5, -0.5))
            creator.create('Individual', nx.DiGraph, fitness=creator.FitnessMulti, statistics=dict)
        self._toolbox = base.Toolbox()
        self._toolbox.register('expr', self._create_individual)
        self._toolbox.register('individual', tools.initIterate, creator.Individual, self._toolbox.expr)
        self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register('evaluate', evaluate)
        self._toolbox.register('select', tools.selBest)
        self._toolbox.register('mate', mate, self._concepts, self._variables)
        self._toolbox.register('mutate', mutate, self._concepts, self._variables)

    def _initialize_SEM(self, individual):
        """ Initialises individual with random connections
        Random connections are created separately for measurement and structural model

        :param individual: an instance of individual (networkx graph)
        :return:
        """
        variables = deepcopy(self._variables)
        pool = cycle(self._concepts)
        # measurement model
        for concept in pool:
            if variables:
                observation = random.choice(variables)
                variables.remove(observation)
                individual.add_edge(observation, concept)
            else:
                break
        # Structural model
        concept_tuples = itertools.combinations(self._concepts, 2)
        for pairs in concept_tuples:
            individual.add_edge(pairs[0], pairs[1])

    def _create_individual(self):
        """ Create a individual (an instance of networkx DiGraph)

        :return: Individual
            an instance of networkx DiGraph
        """
        ind = nx.DiGraph()
        ind.add_nodes_from(self._variables)
        ind.add_nodes_from(self._concepts)
        self._initialize_SEM(ind)
        return ind

    def _evolve(self):
        """ Start evolution

        :return: None
        """
        print('Start of evolution')
        self._hof = tools.HallOfFame(10)
        multi = create_multistatistics(self._fit_indices)
        pop, log = algorithms.eaSimple(self._pop, toolbox=self._toolbox,
                                       cxpb=self._crossover_rate, mutpb=self._mutation_rate, ngen=self._generation,
                                       stats=multi, halloffame=self._hof, verbose=True)
        save_log(self._fit_indices, log)
        compose_SEM(self._hof)

    def fit(self):
        """ Public method to start evolutionary approach to SEM modelling
        Results will be stored under ../../results

        :return: None
        """
        self._setup_toolbox()
        self._pop = self._toolbox.population(self._pop_size)
        self._evolve()




all = AllRelations(generation=10, pop_size=20)
all.fit()
