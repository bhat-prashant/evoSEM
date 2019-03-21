#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import re
import networkx as nx
import matplotlib.pyplot as plt
from evaluate import *
import random
from copy import deepcopy
from deap.tools import HallOfFame


def extract_model(individual):
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
    fitness = re.findall("\d+\.\d+", fitness_str)
    if fitness:
        fitness = [float(i) for i in fitness]
    return tuple(fitness)


def compose_SEM(individual):
    if isinstance(individual, HallOfFame):
        for i, ind in enumerate(individual):
            fig = plt.figure()
            plt.title('Fitness vales - CFI: {}, TLI: {}, \n AIC: {}, BIC: {}, RMSEA: {}'.
                      format(ind.fitness.values[0], ind.fitness.values[1], ind.fitness.values[2],
                             ind.fitness.values[3], ind.fitness.values[4]))
            nx.draw(ind, with_labels=True)
            plt.savefig('../../results/SEM_{}.png'.format(i), format='PNG')
            # plt.show()
    elif isinstance(individual, nx.DiGraph):
        fig = plt.figure()
        nx.draw(individual, with_labels=True)
        plt.savefig('SEM_n')
        # plt.show()


def evaluate(individual):
    model = extract_model(individual)
    fitness = (0, 0, 0, 0, 1)
    try:
        x = evaluate_SEM(model)
        fitness = extract_fitness(x)
    except:
        pass
    return fitness


def mate(concepts, variables, individual_1, individual_2):
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

#TODO --
def validate_individual(concepts,variables, ind):
    individual = deepcopy(ind)
    # remove redundant edges from measurement model
    for var in variables:
        succ = list(individual.successors(var))
        if individual.out_degree(var) > 1:
            for successor in succ[:-1]:
                individual.remove_edges_from(var, successor)
    # if disconnected, add connection between concepts in structural model
    # if not nx.is_strongly_connected(individual):
    #     for con in concepts:
    #         neighbors = list(individual.neighbors(con))
    #         if individual.out_degree(con) == 0:
    #             succ = list(individual.successors(con))
    #             pred = list(individual.predecessors(con))
    return individual


























