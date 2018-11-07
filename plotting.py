import csv
import matplotlib.pyplot as plt
import numpy as np

run_name = 'test'




def get_test(param_nr):
    """
    retrieving relevant parameter-distinctions
    :param param_nr: order of parameter in params.csv
    :return: boolean-array
    """
    bool_array = []

    with open(run_name + '_params.csv', 'r') as f:

        for i, row in enumerate(csv.reader(f)):
            if row[param_nr] == 'True':
                bool_array.append(True)
            elif row[param_nr] == 'False':
                bool_array.append(False)
            else:
                print('wrong param retrieval')
    return bool_array

def plot(string, tests, param_info=''):
    """
    plots two contrasting plots
    :param string: can be 'mean' or 'best'
    :param tests: boolean list of contrasting test parameters
    """
    plot_helper(string, tests, False, param_info)
    plot_helper(string, tests, True, param_info)


def plot_helper(string, tests, negated, param_info):
    """
    plots two contrasting plots
    :param string: can be 'mean' or 'best'
    :param tests: boolean list of contrasting test parameters
    :param negated: if True: plots the negated tests-list
    """

    plotname = run_name + '_' + string + '_' + param_info + str(not negated)

    if negated:
        tests = [not i for i in tests]

    with open(run_name + '_' + string + '.csv', 'r') as f:
        for i, row in enumerate(csv.reader(f)):
            plt.title(plotname)
            if tests[i]:
                plt.plot([int(float(r)) for r in row])
                plt.xlabel('GA-iterations')
                plt.ylabel('Cost')
        plt.savefig('./plots/' + plotname + '_.png')
        plt.gcf().clear()

if __name__ == '__main__':
    tests = get_test(1)
    plot('best', tests, param_info='Heuristic')
    plot('mean', tests, param_info='Heuristic')


