import matplotlib
matplotlib.use('Agg')

import csv
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys


''' INSERT QUERY '''

# if not using command line input
if len(sys.argv) == 1:
    run_name = 'heuristic_test'
    grid_param = 'scenario'
else:
    grid_param = 'scenario'
    run_name = sys.argv[1] + '[%s, %s]'%(float(sys.argv[2]), float(sys.argv[3]))

''' PLOT FUNCTIONS '''

def get_test(param):
    """
    retrieving relevant parameter-distinctions
    :param param: name of parameter
    :return: boolean-array
    """
    # analysis in which runs the parameter had which value
    distinction = []

    with open('./results/' + run_name + '.pickle', 'rb') as f:
        dict_list = pickle.load(f)
        # retrieve position of the relevant parameter
        position = False
        # and retrieve possible parameter values
        vals = []
        for i, elem in enumerate(dict_list):
            if elem == param:
                position = int(i/2)
                vals = dict_list[i +1]

        # creating cartesian product of the dictionary as 2D-list
        # 1.: create a clean list of all parameters without parameter names
        iteration = []
        # if done with one element
        done = False
        for i, e in enumerate(dict_list):
            if done == False:
                done = True
            else:
                iteration.append(e)
                done = False

        # search all permutation of current grid search
        permutation = []
        for i, row in enumerate(itertools.product(*iteration)):
            permutation.append(row)
            if row[position] ==  vals[0]:
                distinction.append(vals[0])
            elif row[position] ==  vals[1]:
                distinction.append(vals[1])
            else:
                print('wrong param retrieval')
    return distinction, vals, permutation

def plot(parameter, modus='best'):
    """
    plots two contrasting plots
    :param parameter: relevant parameter
    :param modus: can be 'mean' or 'best'
    """
    disctinction, vals, permutation = get_test(parameter)

    plot_helper(modus, disctinction, vals[0], parameter, permutation)
    plot_helper(modus, disctinction, vals[1], parameter, permutation)


def plot_helper(string, tests, value, param_info, permutation):
    """
    plots two contrasting plots
    :param string: can be 'mean' or 'best'
    :param tests: distinctions of contrasting test parameters
    :param value: values of the test parameter
    :param param_info: name of the parameter
    """

    plotname = string + '_' + run_name + '_' + param_info + str(value)

    print('plot:', plotname)
    with open('./results/' + run_name + '_' + string + '.csv', 'r') as f:
        for i, row in enumerate(csv.reader(f)):
            plt.title(plotname)
            if tests[i] == value:
                plt.plot([int(float(r)) for r in row], label=permutation[i])
                plt.xlabel('GA-iterations')
                plt.ylabel('Cost')
        # optional command to show a legend
        plt.legend(prop={'size': 6})
        plt.ylim(50000, 100000)
        plt.xlim(0, 600)
        plt.savefig('./plots/' + plotname + '_.png')
        plt.gcf().clear()

if __name__ == '__main__':
    plot(grid_param, 'best')
    plot(grid_param, 'mean')


