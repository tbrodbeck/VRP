import csv
import itertools
import numpy as np
import pickle
from collections import OrderedDict

from model import VRP


''' HYPERPARAMETERS'''

run = 'test2'
use_grid = True

heuristic_test = [
    'scenario', [1, 2],
    'heuristic', [False, True],
    'pop_size', [12],
    'selection_size', [2],
    'aco_iterations', [10],
    'beta', [1],
    'evap_rate', [.1],
    'beta_evap', [0],
    'crossover_prob', [0.05, 0.2],
    'mutation_prob', [0.05, 0.1],
    'reduce_clusters', [0],
    'kmeans_iterations', [10],
    'squared_dist', [True],
    'time_limit', [600]
]

test2 = [
    'scenario', [1, 2],
    'heuristic', [True],
    'pop_size', [10, 14],
    'selection_size', [2, 4],
    'aco_iterations', [15, 25],
    'beta', [1],
    'evap_rate', [.1],
    'beta_evap', [0],
    'crossover_prob', [0.1, 0.2],
    'mutation_prob', [0.05, 0.1],
    'reduce_clusters', [0, 4],
    'kmeans_iterations', [10, 20],
    'squared_dist', [True],
    'time_limit', [600]
]

input_dict = globals()[run]

def create_dict(list_dict):
    """
    Creating a dict from a list-dict
    """
    # True if done with one key-value pair
    done = False
    name = ''
    dict = {}
    for elem in list_dict:
        if done == False:
            name = elem
            done = True
            continue
        if done == True:
            dict[name] = elem
            done = False
            continue
    return dict

def gridsearch(dict_list, run_name, use_grid):

    dict = create_dict(dict_list)

    # creating cartesian product of the dictionary as 2D-list
    # 1.: create a list of all parameters without dictionary names
    iteration = []
    for i, entry in enumerate(dict):
        values = []
        for value in dict[entry]:
            values.append(value)
        iteration.append(values)

    # init saving-lists
    best = []
    mean = []

    # deploying all permutations
    print('All permutations of test-parameters:')
    nr_of_perms = 0
    permutations = []
    for v in (itertools.product(*iteration)):
        permutations.append(v)
        print(v)
        nr_of_perms += 1
    with open('./results/' + run_name + '.pickle', 'wb') as f:
        pickle.dump(dict_list, f)


    if use_grid == False:
        count = 0
        for v in permutations:
            count += 1
            print('Test', count, 'of', nr_of_perms, ':')
            print(dict_list)
            best_run, mean_run = VRP(scenario=v[0], heuristic=v[1], pop_size=v[2], selection_size=v[3], aco_iterations=v[4], beta=v[5], evap_rate=v[6], beta_evap=v[7], crossover_prob=v[8], mutation_prob=v[9], reduce_clusters=v[10], kmeans_iterations=v[11], squared_dist=v[12], time_limit=v[13])

            # saving results
            best.append(best_run)
            mean.append(mean_run)
        # persisting
        print('Endresults for best:')
        with open('./results/' + run_name + '_best.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            for i in best:
                print(i)
                writer.writerow(i)
        print('Endresults for mean:')
        with open('./results/' + run_name + '_mean.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            for i in mean:
                print(i)
                writer.writerow(i)

    else:
        grid_manager(permutations)

def grid_manager(permutations):
    """
    using a special manager for IKW grid deployment
    :param permutations: all permutations of the input to deploy
    """
    pass


if __name__ == '__main__':
    gridsearch(input_dict, run, use_grid)