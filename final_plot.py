import os
import csv
import matplotlib.pyplot as plt
import numpy as np

modus = 'best'
test_name = 'final_test'

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
    filename = './results_final/' + run_name + '_' + modus + '.csv'


    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            row = []
            for rows in csv.reader(f):
                row.append([float(i) for i in rows])
            results.append(row[0])
    else:
          print('file not found')

#print(results)

# for every scenario
for scenario in range(2):
    scenario += 1
    values = []
    for run in range(10):
        if scenario == 2:
            run += 10

        #print('plotting ' + str(results[run]))
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




