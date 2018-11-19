import sys

from grid_search import gridsearch


# parsing command-line imputs
parameter = sys.argv[1]
value1 = float(sys.argv[2])
value2 = float(sys.argv[3])


dict_list = [
    'scenario', [1, 2],
    'heuristic', [True],
    'pop_size', [14],
    'selection_size', [4],
    'aco_iterations', [15],
    'beta', [1],
    'evap_rate', [.1],
    'beta_evap', [0.2],
    'crossover_prob', [0.1],
    'mutation_prob', [0.1],
    'reduce_clusters', [4],
    'kmeans_iterations', [20],
    'squared_dist', [False],
    'time_limit', [20] # in minutes
]

def search_init(dict_list, parameter, test_values):
    """
    searches
    :param dict_list: standa.rd parameters
    :param parameter: param we want to analyze
    :param test_values: how we want to change the parameter
    """
    for i, e in enumerate(dict_list):
        if e == parameter:
            print(e, i)
            dict_list[i + 1] = test_values

    gridsearch(dict_list, parameter + str(test_values))


search_init(dict_list, parameter, [value1, value2])

