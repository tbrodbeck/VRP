import os
import sys


print('cwd', os.getcwd())
run = int(sys.argv[1])


command = 'python3 ikw_grid_deploy.py '
command2 = 'python3 plotting.py '

if run == 1:
	input = 'pop_size 12 16'

if run == 2:
	input = 'selection_size 3 6'

if run == 3:
	input = 'aco_iterations 10 20'

if run == 4:
	input = 'beta 0.99 1'

if run == 5:
	input = 'evap_rate .1 .3'

if run == 6:
	input = 'crossover_prob 0.05 0.15'

if run == 7:
	input = 'mutation_prob 0.07 0.13'

if run == 8:
	input = 'reduce_clusters 2 6'

if run == 9:
	input = 'kmeans_iterations 20 30'

if run == 10:
	input = 'squared_dist 0 1'

os.system(command + input)
os.system(command2 + input)

