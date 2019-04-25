#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import re
import networkx as nx
import matplotlib.pyplot as plt
from evaluate import *
import random
from copy import deepcopy
from deap.tools import HallOfFame, Statistics, MultiStatistics
import itertools
import numpy as np
import pandas as pd

def extract_model(individual):
    """ Convert netwrokx graph to 'lavaan' specific SEM model string

    :param individual: an instance of individual (networkx graph)
    :return: string,
        lavaan specific model description
    """
    measurement_model = []
    structural_model = []
    for node in individual._pred:
        pred = individual._pred[node]
        if pred:
            predecessors = list(pred.keys())
            measurement_var = [item for item in predecessors if item.startswith('x')]
            structural_var = [item for item in predecessors if item.startswith('c')]
            if measurement_var:
                separator = ' + '
                measurement_entry = node + ' =~ ' + separator.join(measurement_var)
                measurement_model.append(measurement_entry)
            if structural_var:
                separator = ' + '
                structural_entry = node + ' ~ ' + separator.join(structural_var)
                structural_model.append(structural_entry)
    measurement_model = '\n'.join(measurement_model)
    structural_model = '\n'.join(structural_model)
    model = measurement_model + '\n' + structural_model
    return model


def extract_fitness(fitness_str):
    """ Parse results from fitness string

    :param fitness_str: String
        String received from R
    :return: tuple
        fitness values as a tuple
    """
    fitness = re.findall("\d+\.\d+", fitness_str)
    if fitness and 'warning' not in fitness_str:
        fitness = [float(i) for i in fitness]
    else:
        raise Exception('Warning occured')
    return tuple(fitness)


def compose_SEM(individual):
    """ Plot / save an individual (networkx graph) using plt
    Future Work: make graph generation dynamic

    :param individual: an instance of individual (networkx graph)
    :return: None
    """
    if isinstance(individual, HallOfFame):
        for i, ind in enumerate(individual):
            fig = plt.figure()
            plt.title('Fitness vales - CFI: {}, TLI: {}, \n AIC: {}, BIC: {}, RMSEA: {}'.
                      format(ind.fitness.values[0], ind.fitness.values[1], ind.fitness.values[2],
                             ind.fitness.values[3], ind.fitness.values[4]))
            values = [0.25 if 'x' in node else 0.75 for node in ind.nodes()]
            nx.draw_circular(ind, with_labels=True, font_color='white', node_size=500, cmap=plt.get_cmap('RdBu'), node_color=values)
            plt.savefig('../../results/SEM_{}.png'.format(i), format='PNG')
            # plt.show()
    elif isinstance(individual, nx.DiGraph):
        fig = plt.figure()
        nx.draw(individual, with_labels=True)
        plt.savefig('SEM_n')
        # plt.show()


def evaluate(individual):
    """ Evaluation function for evolution

    :param individual: an instance of individual (networkx graph)
    :return: tuple
        fitness values as tuple
    """
    model = extract_model(individual)
    fitness = (0, 0, 0, 0, 1)
    try:
        x = evaluate_SEM(model)
        fitness = extract_fitness(x)
    except:
        pass
    return fitness


def mate(concepts, variables, individual_1, individual_2):
    """ Mate operator for evolution
    Switches structural model between two individuals

    :param concepts: list
        list of strings as probable concepts
    :param variables: list
        list of variables in the dataset
    :param individual_1: an instance of individual (networkx graph)
    :param individual_2: an instance of individual (networkx graph)
    :return: tuple
        tuple of two new offsprings (an instance of individuals (networkx graph))
    """
    ind_1 = deepcopy(individual_1)
    ind_2 = deepcopy(individual_2)
    c_1 = individual_1.subgraph(concepts)
    c_2 = individual_2.subgraph(concepts)
    ind_1.remove_edges_from(c_1.edges())
    ind_1.add_edges_from(c_2.edges())
    ind_2.remove_edges_from(c_2.edges())
    ind_2.add_edges_from(c_1.edges())
    ind_1 = validate_individual(concepts, variables, ind_1)
    ind_2 = validate_individual(concepts, variables, ind_2)
    return ind_1, ind_2


def mutate(concepts,variables, ind):
    """ Mutate operator for evolution
    Between two randomly chosen concepts, exchanges two variables between concepts.
    and reverses direction between those concepts

    :param concepts: list
        list of strings as probable concepts
    :param variables: list
        list of variables in the dataset
    :param ind: an instance of individual (networkx graph)
    :return: tuple
        tuple of one offspring (an instance of individual (networkx graph))
    """
    individual = deepcopy(ind)
    concept_1 = random.choice(concepts)
    concept_2 = random.choice(concepts)
    if concept_1 != concept_2:
        pred_1 = list(individual._pred[concept_1].keys())
        pred_2 = list(individual._pred[concept_2].keys())
        pred_var_1 = [item for item in pred_1 if item.startswith('x')]
        pred_var_2 = [item for item in pred_2 if item.startswith('x')]
        if pred_var_1 and pred_var_2:
            # exchange the edges between concepts
            var_1 = random.choice(pred_var_1)
            var_2 = random.choice(pred_var_2)
            individual.remove_edge(var_1, concept_1)
            individual.remove_edge(var_2, concept_2)
            individual.add_edge(var_1, concept_2)
            individual.add_edge(var_2, concept_1)
        pred_concept_1 = [item for item in pred_1 if item.startswith('c')]
        if pred_concept_1:
            # reverse the direction of the edge between concepts
            individual.remove_edge(pred_concept_1[0], concept_1)
            individual.add_edge(concept_1, pred_concept_1[0])
    individual = validate_individual(concepts, variables, individual)
    return individual,


def validate_individual(concepts,variables, ind):
    """ Validates individual
    Verify whether there is at least and at most one edge between all nodes

    :param concepts: list
        list of strings as probable concepts
    :param variables: list
        list of variables in the dataset
    :param ind: an instance of individual (networkx graph)
    :return: an instance of individual (networkx graph)
    """
    individual = deepcopy(ind)
    # remove redundant edges from measurement model
    for var in variables:
        succ = list(individual.successors(var))
        if individual.out_degree(var) > 1:
            for successor in succ[:-1]:
                individual.remove_edges_from(var, successor)

    # at least and at most one edge between concepts
    concept_tuples = itertools.combinations(concepts, 2)
    for pairs in concept_tuples:
        edges = individual.number_of_edges(pairs[0], pairs[1]) + individual.number_of_edges(pairs[1], pairs[0])
        if edges > 1:
            individual.remove_edge(pairs[0], pairs[1])
        elif edges < 1:
            individual.add_edge(pairs[0], pairs[1])

    return individual


def create_multistatistics(fitness_indices):
    """ create multi statistics for evolution

    :param fitness_indices: list
        list of strings representing fit indices. These should be in same order as used during evolution
    :return: an instance of deap.tools.MultiStatistics
    """
    statistics = {}
    for i, fit_index in enumerate(fitness_indices):
        stats = create_statistics(fit_index, i)
        statistics[fit_index] = stats
    multi = MultiStatistics(statistics)
    return multi

def create_statistics(fit_index, index):
    """ Create deap.tools.Statistics instance for the given name and index

    :param fit_index: string
        name for the statistic
    :param index: int
        This is the index of the statistic in a multi-objective fitness tuple
    :return: deap.tools.Statistics
    """
    stats = Statistics(lambda ind: ind.fitness.values[index])
    stats.register("{}_avg".format(fit_index), np.mean)
    stats.register("{}_std".format(fit_index), np.std)
    stats.register("{}_min".format(fit_index), np.min)
    stats.register("{}_max".format(fit_index), np.max)
    return stats

def save_log(fitness_indices, logbook):
    """ Save evolutionary log to csv

    :param fitness_indices: list
        List of strings representing fitnesses in a multi objective fitness scenario
    :param logbook:  an instance of deap.tools.Logbook
    :return: None
    """
    for fit_index in fitness_indices:
        fit_df = pd.DataFrame(logbook.chapters[fit_index])
        generation_df = pd.DataFrame(logbook)
        result = pd.concat([fit_df, generation_df],  sort=False, axis=1)
        result.to_csv('../../results/{}_log.csv'.format(fit_index))


def plot_logbook(logbook):
    gen = logbook.select("gen")
    fit_mins = logbook.chapters["cfi"].select("min")
    size_avgs = logbook.chapters["cfi"].select("max")
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("CFI", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, size_avgs, "r-", label="Maxiimum Fitness")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()













