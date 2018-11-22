import os
import sys

""" 
Final search to be deployed on the ikw grid.
It is designed for 20 parallel runs.
"""

print('cwd', os.getcwd())
run = int(sys.argv[1])


command = 'python3 search.py '
command2 = 'python3 plotting.py '

# choose scenario
if run <= 10:
    input = '1 ' + str(run)
elif run <= 20:
    input = '2 ' + str(run)

os.system(command + input)

