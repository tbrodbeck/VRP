import os
import csv
import matplotlib.pyplot as plt
import numpy as np

"""
This script plots a representation of a array of runs.
The configuration is 10 plots for scenario 1 and 10 more plots for scenario 2.
"""

''' hyperparameters'''

modus = 'best'
test_name = 'final_test'

''' data sorting '''
results = []
run_names = []
# plot our 20 final runs
for run in range(20):
    run += 1
    scenario = 0
    if run <= 10:
        scenario = 1
    elif run <= 20:
        scenario = 2

    run_name = test_name + str(scenario) + str(run)
    run_names.append(run_name)
    filename = './results/' + run_name + '_' + modus + '.csv'


    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            row = []
            for rows in csv.reader(f):
                row.append([float(i) for i in rows])
            results.append(row[0])
    else:
          print('file not found')

''' plotting '''
# for every scenario
for scenario in range(2):
    scenario += 1
    values = []
    for run in range(10):
        if scenario == 2:
            run += 10

        plt.plot(results[run], label='Run %s'%(run+1))
        plt.xlabel('GA-iterations')
        plt.ylabel('Cost')
        plt.title('Scenario %s'%(scenario))

        values.append(results[run][-1])

    plt.legend(prop={'size': 6})
    plt.savefig('./plots/' + run_names[run] + '_.png')
    plt.gcf().clear()

    print('scenario:', scenario)
    print('mean', np.mean(values))
    print('stderror', np.std(values))




