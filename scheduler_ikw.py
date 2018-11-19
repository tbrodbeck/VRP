import os
import sys


print('cwd', os.getcwd())
run = int(sys.argv[1])


command = 'python3 ikw_grid_deploy.py '
command2 = 'python3 plotting.py '

if run == 1:
	input = 'pop_size 14 16'

if run == 2:
	input = 'selection_size 6 8'

if run == 3:
	input = 'aco_iterations 15 20'

if run == 4:
	input = 'beta 0.95 0.99'

if run == 5:
	input = 'evap_rate .05 .15'

if run == 6:
	input = 'beta_evap 0 0.2'

if run == 7:
	input = 'crossover_prob 0.05 0.1'

if run == 8:
	input = 'mutation_prob 0.05 0.1'

if run == 9:
	input = 'reduce_clusters 4 8'

if run == 10:
	input = 'kmeans_iterations 15 25'

if run == 11:
	input = 'squared_dist 0 1'

os.system(command + input)
os.system(command2 + input)

